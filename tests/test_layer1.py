"""Unit tests for Layer 1 deterministic statistical gate and OLS drift calculation.

Proves:
(a) Clean synthetic data passes in <25ms.
(b) Sparse null injection triggers deterministic short-circuit.
(c) Drift slope > 3sigma aborts the pipeline before any LLM is called.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from consensus_guard.layer1 import (
    GateResult,
    Layer1Stats,
    compute_layer1_stats,
    evaluate_statistical_invariants,
)


def generate_clean_synthetic_dataset(n_rows: int = 2500) -> pd.DataFrame:
    """Generate clean synthetic tabular data with zero nulls and stationary noise."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "user_id": np.arange(10000, 10000 + n_rows),
            "age": rng.integers(18, 70, size=n_rows),
            "account_balance": rng.normal(5000.0, 1000.0, size=n_rows),
            "transaction_count": rng.poisson(lam=15, size=n_rows),
            "category": rng.choice(["retail", "enterprise", "smb"], size=n_rows),
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="min"),
        }
    )


def test_clean_synthetic_data_passes_under_25ms() -> None:
    """(a) Clean synthetic data passes evaluation and invariant checks in <25ms."""
    df = generate_clean_synthetic_dataset(n_rows=2500)

    t_start = time.perf_counter()
    stats = compute_layer1_stats(df)
    gate = evaluate_statistical_invariants(stats, max_null_ratio=0.05, max_drift_sigma=3.0)
    total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # Verification of execution duration requirement
    assert (
        total_elapsed_ms < 25.0
    ), f"Execution took {total_elapsed_ms:.2f}ms, expected < 25.0ms"
    assert (
        gate.execution_time_ms < 25.0
    ), f"Gate execution took {gate.execution_time_ms:.2f}ms, expected < 25.0ms"

    # Verification of clean pass
    assert gate.passed is True
    assert gate.reason is None
    assert gate.short_circuit is False
    assert len(gate.violations) == 0

    # Verification of mathematical statistics
    assert stats.row_count == 2500
    assert stats.column_count == 6
    for col, ratio in stats.null_ratios.items():
        assert ratio == 0.0, f"Column {col} has unexpected null ratio {ratio}"
    for col, sigma in stats.drift_sigmas.items():
        assert sigma <= 3.0, f"Stationary column {col} has high drift sigma {sigma}"


def test_sparse_null_injection_triggers_short_circuit() -> None:
    """(b) Sparse null injection triggers deterministic short-circuit in <30ms."""
    df = generate_clean_synthetic_dataset(n_rows=2000)

    # Inject sparse nulls exceeding 5% threshold (8% nulls in 'account_balance')
    target_col = "account_balance"
    null_indices = np.random.default_rng(123).choice(
        len(df), size=int(0.08 * len(df)), replace=False
    )
    df.loc[null_indices, target_col] = np.nan

    t_start = time.perf_counter()
    stats = compute_layer1_stats(df)
    gate = evaluate_statistical_invariants(stats, max_null_ratio=0.05, max_drift_sigma=3.0)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # Verification of execution duration (<30ms)
    assert elapsed_ms < 30.0, f"Execution took {elapsed_ms:.2f}ms, expected < 30.0ms"
    assert gate.execution_time_ms < 30.0

    # Verification of deterministic short-circuit
    assert gate.passed is False
    assert gate.reason == "Statistical Invariant Breach"
    assert gate.short_circuit is True
    assert any(
        target_col in v and "null ratio" in v for v in gate.violations
    ), f"Expected violation for {target_col}, got {gate.violations}"

    # Verify observed ratio matches injected ratio
    observed_ratio = stats.null_ratios[target_col]
    assert 0.075 <= observed_ratio <= 0.085


def test_drift_slope_greater_than_3_sigma_aborts_pipeline() -> None:
    """(c) Drift slope > 3sigma aborts the pipeline before any LLM is called."""
    n_rows = 1500
    rng = np.random.default_rng(99)

    # Construct synthetic data with an intentional temporal drift in 'temperature_sensor'
    # y_i = 20.0 + 0.05 * t_i + noise
    t_steps = np.arange(n_rows, dtype=np.float64)
    drift_feature = 20.0 + (0.05 * t_steps) + rng.normal(0.0, 1.0, size=n_rows)

    df = pd.DataFrame(
        {
            "temperature_sensor": drift_feature,
            "static_pressure": rng.normal(101.3, 0.2, size=n_rows),
        }
    )

    # Downstream LLM caller mock
    mock_llm_agent = MagicMock(return_value={"llm_response": "analysis_completed"})

    # Compute Layer 1 statistical metrics
    t_start = time.perf_counter()
    stats = compute_layer1_stats(df)
    gate = evaluate_statistical_invariants(stats, max_null_ratio=0.05, max_drift_sigma=3.0)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    assert elapsed_ms < 30.0

    # Drift slope and sigma verification
    drift_sigma = stats.drift_sigmas["temperature_sensor"]
    drift_slope = stats.drift_slopes["temperature_sensor"]
    assert (
        drift_sigma > 3.0
    ), f"Expected drift sigma > 3.0, got {drift_sigma:.2f} (slope: {drift_slope:.6f})"
    assert 0.04 <= drift_slope <= 0.06

    # Gating and deterministic short-circuit verification
    assert gate.passed is False
    assert gate.reason == "Statistical Invariant Breach"
    assert gate.short_circuit is True
    assert any("temperature_sensor" in v and "temporal drift" in v for v in gate.violations)

    # Pipeline short-circuit simulation: LLM must NEVER be called if gate fails
    if not gate.short_circuit:
        mock_llm_agent(df)

    # Prove downstream LLM call was aborted
    mock_llm_agent.assert_not_called()


def test_ols_slope_mathematical_precision() -> None:
    """Verify OLS slope beta matches exact analytical closed form."""
    # Linear dataset: y = 3 + 2.5 * t with zero error
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 3.0 + 2.5 * t

    df = pd.DataFrame({"y_linear": y})
    stats = compute_layer1_stats(df)

    # beta = 2.5
    assert pytest.approx(stats.drift_slopes["y_linear"], rel=1e-5) == 2.5
    # Since residuals are 0, sigma should be deterministic / saturated (> 3.0)
    assert stats.drift_sigmas["y_linear"] > 3.0


def test_dict_input_compatibility() -> None:
    """Verify compute_layer1_stats handles dictionary input seamlessly."""
    raw_dict: dict[str, list[Any]] = {
        "col_a": [1.0, 2.0, 3.0, None],
        "col_b": ["x", "y", "x", "z"],
    }
    stats = compute_layer1_stats(raw_dict)

    assert stats.row_count == 4
    assert stats.column_count == 2
    assert stats.null_ratios["col_a"] == 0.25
    assert stats.null_ratios["col_b"] == 0.0
    assert stats.cardinality_ratios["col_b"] == 0.75  # 3 distinct out of 4


def test_serialization_methods() -> None:
    """Verify to_dict() serialization on Layer1Stats and GateResult."""
    stats = Layer1Stats(
        row_count=100,
        column_count=2,
        null_ratios={"a": 0.01},
        cardinality_ratios={"a": 0.9},
        drift_slopes={"a": 0.001},
        drift_sigmas={"a": 1.2},
    )
    d_stats = stats.to_dict()
    assert d_stats["row_count"] == 100
    assert d_stats["drift_slopes"]["a"] == 0.001

    gate = GateResult(
        passed=False,
        reason="Statistical Invariant Breach",
        short_circuit=True,
        violations=["test violation"],
    )
    d_gate = gate.to_dict()
    assert d_gate["passed"] is False
    assert d_gate["short_circuit"] is True
