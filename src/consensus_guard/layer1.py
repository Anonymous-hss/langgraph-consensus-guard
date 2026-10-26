"""Layer 1 deterministic statistical gate and OLS drift calculation.

Provides high-speed (<30ms) mathematical invariant verification for tabular data
prior to any LLM invocation, establishing an immutable ground-truth barrier against
hallucinated consensus and corrupted pipelines.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Layer1Stats:
    """Statistical summary metrics evaluated deterministically on data artifacts."""

    row_count: int
    column_count: int
    null_ratios: dict[str, float]
    cardinality_ratios: dict[str, float]
    drift_slopes: dict[str, float]
    drift_sigmas: dict[str, float]
    drift_std_errors: dict[str, float] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert Layer1Stats instance to a plain dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "null_ratios": self.null_ratios,
            "cardinality_ratios": self.cardinality_ratios,
            "drift_slopes": self.drift_slopes,
            "drift_sigmas": self.drift_sigmas,
            "drift_std_errors": self.drift_std_errors,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class GateResult:
    """Outcome of statistical invariant gating."""

    passed: bool
    reason: str | None = None
    short_circuit: bool = False
    violations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert GateResult instance to a plain dictionary."""
        return {
            "passed": self.passed,
            "reason": self.reason,
            "short_circuit": self.short_circuit,
            "violations": self.violations,
            "details": self.details,
            "execution_time_ms": self.execution_time_ms,
        }


def compute_layer1_stats(
    df: pd.DataFrame | dict[str, Any],
    time_col: str | None = None,
    exclude_drift_cols: list[str] | set[str] | None = None,
) -> Layer1Stats:
    """Compute Layer 1 deterministic statistics on a tabular dataset.

    Calculates:
    1. Column-level null ratios: count(null) / N.
    2. Cardinality ratios: distinct(col) / N.
    3. Temporal drift via Ordinary Least Squares (OLS) slope on numeric features:
       beta = sum((t_i - bar{t}) * (y_i - bar{y})) / sum((t_i - bar{t})^2).
       and the corresponding standard error and t-statistic (drift sigma).

    Parameters:
        df: Input pandas DataFrame or dictionary representing column vectors or records.
        time_col: Optional column name to use as explicit temporal index t. If omitted,
            auto-detects datetime columns or falls back to row sequence index t_i = 0..N-1.
        exclude_drift_cols: Optional list or set of column names to exclude from temporal drift
            evaluation (e.g. non-feature surrogate keys).

    Returns:
        Layer1Stats containing mathematical measurements and calculation duration in ms.
    """
    t_start = time.perf_counter()
    exclude_set = set(exclude_drift_cols) if exclude_drift_cols else set()

    # Normalize input into pandas DataFrame
    if isinstance(df, pd.DataFrame):
        data = df
    else:
        try:
            data = pd.DataFrame(df)
        except ValueError:
            data = pd.DataFrame([df])

    n_rows = len(data)
    columns = list(data.columns)
    n_cols = len(columns)

    if n_rows == 0:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return Layer1Stats(
            row_count=0,
            column_count=n_cols,
            null_ratios=dict.fromkeys(columns, 0.0),
            cardinality_ratios=dict.fromkeys(columns, 0.0),
            drift_slopes={},
            drift_sigmas={},
            drift_std_errors={},
            execution_time_ms=round(elapsed_ms, 3),
            metadata={"empty": True},
        )

    # 1. Column-level null ratios: count(null) / N
    null_counts = data.isna().sum()
    null_ratios: dict[str, float] = {
        col: round(float(null_counts[col] / n_rows), 6) for col in columns
    }

    # 2. Cardinality ratio: distinct(col) / N
    nunique_counts = data.nunique(dropna=False)
    cardinality_ratios: dict[str, float] = {
        col: round(float(nunique_counts[col] / n_rows), 6) for col in columns
    }

    # 3. Temporal drift via OLS slope on numeric features
    # Determine temporal coordinate vector t
    explicit_time = False
    if time_col and time_col in data.columns:
        explicit_time = True
        t_raw = data[time_col]
        if pd.api.types.is_datetime64_any_dtype(t_raw):
            t_num = pd.to_numeric(pd.to_datetime(t_raw), errors="coerce")
            t_vals = np.asarray(t_num, dtype=np.float64) / 1e9
        else:
            t_vals = np.asarray(pd.to_numeric(t_raw, errors="coerce"), dtype=np.float64)
    else:
        # Check for auto-detectable datetime columns
        datetime_cols = [
            c for c in columns if pd.api.types.is_datetime64_any_dtype(data[c])
        ]
        if datetime_cols:
            explicit_time = True
            time_col = datetime_cols[0]
            t_num = pd.to_numeric(pd.to_datetime(data[time_col]), errors="coerce")
            t_vals = np.asarray(t_num, dtype=np.float64) / 1e9
        else:
            t_vals = np.arange(n_rows, dtype=np.float64)

    drift_slopes: dict[str, float] = {}
    drift_sigmas: dict[str, float] = {}
    drift_std_errors: dict[str, float] = {}

    # Identify numeric candidate feature columns for drift
    for col in columns:
        if explicit_time and col == time_col:
            continue
        col_lower = col.lower()
        if (
            (col in exclude_set)
            or col_lower in {"id", "uuid", "pk", "key", "index"}
            or col_lower.endswith(("_id", "_uuid", "_key", "_pk"))
        ):
            continue

        col_series = data[col]
        if not pd.api.types.is_numeric_dtype(col_series):
            continue

        y_vals = np.asarray(pd.to_numeric(col_series, errors="coerce"), dtype=np.float64)
        valid_mask: np.ndarray[Any, Any] = ~np.isnan(y_vals) & ~np.isnan(t_vals)
        t_sub: np.ndarray[Any, Any] = t_vals[valid_mask]
        y_sub: np.ndarray[Any, Any] = y_vals[valid_mask]
        sub_n = len(y_sub)

        if sub_n < 2:
            drift_slopes[col] = 0.0
            drift_sigmas[col] = 0.0
            drift_std_errors[col] = 0.0
            continue

        t_bar = float(np.mean(t_sub))
        y_bar = float(np.mean(y_sub))
        dt = t_sub - t_bar
        dy = y_sub - y_bar
        ss_tt = float(np.sum(dt**2))
        ss_ty = float(np.sum(dt * dy))

        if ss_tt <= 1e-15:
            drift_slopes[col] = 0.0
            drift_sigmas[col] = 0.0
            drift_std_errors[col] = 0.0
            continue

        beta = ss_ty / ss_tt
        residuals = dy - (beta * dt)
        sse = float(np.sum(residuals**2))
        df_resid = max(1, sub_n - 2)
        s2 = sse / df_resid
        se_beta = math.sqrt(max(0.0, s2 / ss_tt))

        if se_beta > 1e-12:
            sigma = abs(beta) / se_beta
        elif abs(beta) > 1e-12:
            sigma = 999.0  # Perfect deterministic drift without noise
        else:
            sigma = 0.0

        drift_slopes[col] = round(float(beta), 8)
        drift_sigmas[col] = round(float(sigma), 4)
        drift_std_errors[col] = round(float(se_beta), 8)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return Layer1Stats(
        row_count=n_rows,
        column_count=n_cols,
        null_ratios=null_ratios,
        cardinality_ratios=cardinality_ratios,
        drift_slopes=drift_slopes,
        drift_sigmas=drift_sigmas,
        drift_std_errors=drift_std_errors,
        execution_time_ms=round(elapsed_ms, 3),
        metadata={"time_col_used": time_col if explicit_time else "row_index"},
    )


def evaluate_statistical_invariants(
    stats: Layer1Stats,
    max_null_ratio: float = 0.05,
    max_drift_sigma: float = 3.0,
) -> GateResult:
    """Evaluate statistical invariants against computed Layer 1 measurements.

    If any statistical invariants fail (null ratio > max_null_ratio or
    temporal drift > max_drift_sigma), short-circuits the pipeline immediately
    in <30ms, preventing any LLM or downstream agent invocation.

    Parameters:
        stats: Computed Layer1Stats object.
        max_null_ratio: Maximum permissible fraction of null values per column (default 0.05).
        max_drift_sigma: Maximum permissible temporal drift slope significance (default 3.0 sigma).

    Returns:
        GateResult indicating whether invariants passed or were breached.
    """
    t_start = time.perf_counter()
    violations: list[str] = []
    details: dict[str, Any] = {
        "max_null_ratio_threshold": max_null_ratio,
        "max_drift_sigma_threshold": max_drift_sigma,
        "failed_null_columns": {},
        "failed_drift_columns": {},
    }

    # 1. Null ratio invariant verification
    for col, null_ratio in stats.null_ratios.items():
        if null_ratio > max_null_ratio:
            violations.append(
                f"Column '{col}' null ratio {null_ratio:.4f} exceeds threshold {max_null_ratio:.4f}"
            )
            details["failed_null_columns"][col] = null_ratio

    # 2. Temporal drift invariant verification
    for col, drift_sigma in stats.drift_sigmas.items():
        if drift_sigma > max_drift_sigma:
            slope = stats.drift_slopes.get(col, 0.0)
            violations.append(
                f"Column '{col}' temporal drift slope {slope:+.6f} ({drift_sigma:.2f} sigma) "
                f"exceeds invariant threshold {max_drift_sigma:.2f} sigma"
            )
            details["failed_drift_columns"][col] = {
                "drift_sigma": drift_sigma,
                "drift_slope": slope,
            }

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if violations:
        return GateResult(
            passed=False,
            reason="Statistical Invariant Breach",
            short_circuit=True,
            violations=violations,
            details=details,
            execution_time_ms=round(elapsed_ms, 3),
        )

    return GateResult(
        passed=True,
        reason=None,
        short_circuit=False,
        violations=[],
        details=details,
        execution_time_ms=round(elapsed_ms, 3),
    )
