"""Tests for OpenTelemetry instrumentation in Consensus Guard.

Verifies:
1. Layer 1 statistical rejection records:
   - consensus_guard.layer1.short_circuited == True
   - consensus_guard.tokens_saved_estimate > 0
   - consensus_guard.layer1.max_null_ratio
   - consensus_guard.layer1.max_ols_drift_sigma
   without invoking downstream agent nodes.
2. Clean passing datasets invoke downstream voter fanout and consensus reducer:
   - consensus_guard.consensus.voter_variance (float)
   - consensus_guard.consensus.judge_override (bool)
   - tokens_saved_estimate == 0
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import opentelemetry.trace as trace
import pandas as pd
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from consensus_guard.graph import build_consensus_graph
from consensus_guard.telemetry import (
    consensus_reducer,
    domain_voters_fanout,
    estimate_tokens_saved,
    layer1_gate,
)
from consensus_guard.voters import DomainVote

# Global in-memory exporter initialized once for OpenTelemetry provider singleton
_GLOBAL_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_GLOBAL_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


@pytest.fixture(autouse=True)
def otel_exporter() -> InMemorySpanExporter:
    """Fixture that resets the in-memory span exporter before every test."""
    _GLOBAL_EXPORTER.clear()
    return _GLOBAL_EXPORTER


def test_layer1_statistical_rejection_records_telemetry_without_downstream(
    otel_exporter: InMemorySpanExporter,
) -> None:
    """Assert Layer 1 rejection records short_circuited=True and tokens_saved_estimate > 0.

    Crucially verifies that downstream agent nodes (domain_voters_fanout, consensus_reducer)
    are NOT invoked.
    """
    n = 200
    # Inject severe nulls in 'feature_a' (30% nulls vs 5% threshold)
    feature_a = np.random.randn(n)
    feature_a[:60] = np.nan

    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "feature_a": feature_a,
            "feature_b": np.random.randn(n),
        }
    )

    graph = build_consensus_graph()
    state = asyncio.run(graph.ainvoke({"raw_data": df}))

    # Verify pipeline state short-circuited
    assert state["short_circuited"] is True
    assert state["consensus_verdict"] == "REJECTED"
    assert len(state["votes"]) == 0

    # Retrieve all finished spans from OpenTelemetry exporter
    spans = otel_exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    # Verify Layer 1 Gate span was recorded
    l1_spans = [s for s in spans if "layer1_gate" in s.name]
    assert len(l1_spans) == 1, f"Expected 1 layer1_gate span, found: {span_names}"
    l1_span = l1_spans[0]

    # Check explicit required span attributes
    attrs = l1_span.attributes or {}
    assert (
        attrs.get("consensus_guard.layer1.short_circuited") is True
    ), "Expected consensus_guard.layer1.short_circuited == True"

    tokens_saved = attrs.get("consensus_guard.tokens_saved_estimate")
    assert isinstance(tokens_saved, int), f"tokens_saved_estimate must be int, got {type(tokens_saved)}"
    assert tokens_saved > 0, f"Expected tokens_saved_estimate > 0, got {tokens_saved}"

    max_null = attrs.get("consensus_guard.layer1.max_null_ratio")
    assert isinstance(max_null, (int, float)), f"max_null_ratio must be float, got {type(max_null)}"
    assert float(max_null) >= 0.25, f"Expected max_null_ratio >= 0.25, got {max_null}"

    max_drift = attrs.get("consensus_guard.layer1.max_ols_drift_sigma")
    assert isinstance(max_drift, (int, float)), f"max_ols_drift_sigma must be float, got {type(max_drift)}"

    # CRUCIAL ASSERTION: downstream agent nodes must NOT have been invoked
    downstream_spans = [
        s for s in spans if "domain_voters" in s.name or "consensus_reducer" in s.name
    ]
    assert (
        len(downstream_spans) == 0
    ), f"Downstream agent nodes were unexpectedly invoked: {[s.name for s in downstream_spans]}"


def test_layer1_drift_rejection_records_telemetry(
    otel_exporter: InMemorySpanExporter,
) -> None:
    """Assert Layer 1 temporal drift (>3 sigma) rejection records telemetry and short-circuits."""
    n = 300
    t = np.arange(n, dtype=np.float64)
    # Strong deterministic monotonic drift slope
    drifting_feature = 10.0 + (0.5 * t) + np.random.normal(0, 0.1, size=n)

    df = pd.DataFrame(
        {
            "sensor_temp": drifting_feature,
            "pressure": np.random.normal(100.0, 1.0, size=n),
        }
    )

    graph = build_consensus_graph()
    state = asyncio.run(graph.ainvoke({"raw_data": df}))

    assert state["short_circuited"] is True

    spans = otel_exporter.get_finished_spans()
    l1_spans = [s for s in spans if "layer1_gate" in s.name]
    assert len(l1_spans) == 1
    l1_span = l1_spans[0]

    attrs = l1_span.attributes or {}
    assert attrs.get("consensus_guard.layer1.short_circuited") is True
    assert attrs.get("consensus_guard.tokens_saved_estimate") is not None
    tokens_saved = attrs.get("consensus_guard.tokens_saved_estimate")
    assert isinstance(tokens_saved, int) and tokens_saved > 0

    max_drift = attrs.get("consensus_guard.layer1.max_ols_drift_sigma")
    assert isinstance(max_drift, (int, float))
    assert float(max_drift) > 3.0

    # Ensure no voter or reducer spans exist
    assert not any("domain_voters" in s.name for s in spans)
    assert not any("consensus_reducer" in s.name for s in spans)


def test_clean_pipeline_records_downstream_variance_and_override_telemetry(
    otel_exporter: InMemorySpanExporter,
) -> None:
    """Assert passing data executes fanout and records voter_variance and judge_override."""
    rng = np.random.default_rng(101)
    n = 250
    df = pd.DataFrame(
        {
            "account_id": np.arange(1000, 1000 + n),
            "balance": rng.normal(5000.0, 200.0, size=n),
            "status": rng.choice(["active", "verified"], size=n),
        }
    )

    graph = build_consensus_graph()
    state = asyncio.run(graph.ainvoke({"raw_data": df}))

    assert state["short_circuited"] is False
    assert state["consensus_verdict"] == "APPROVED"

    spans = otel_exporter.get_finished_spans()
    span_map = {s.name: s for s in spans}

    # 1. Layer 1 span
    l1_span = span_map.get("consensus_guard.layer1_gate")
    assert l1_span is not None
    assert l1_span.attributes is not None
    assert l1_span.attributes["consensus_guard.layer1.short_circuited"] is False
    assert l1_span.attributes["consensus_guard.tokens_saved_estimate"] == 0

    # 2. Downstream Domain Voters Fan-Out span
    fanout_span = span_map.get("consensus_guard.domain_voters_fanout")
    assert fanout_span is not None
    assert fanout_span.attributes is not None
    v_count = fanout_span.attributes.get("consensus_guard.voter_count")
    assert isinstance(v_count, int) and v_count >= 2

    # 3. Consensus Reducer span
    reducer_span = span_map.get("consensus_guard.consensus_reducer")
    assert reducer_span is not None
    assert reducer_span.attributes is not None

    voter_variance = reducer_span.attributes.get("consensus_guard.consensus.voter_variance")
    assert isinstance(voter_variance, (int, float))
    assert float(voter_variance) >= 0.0

    judge_override = reducer_span.attributes.get("consensus_guard.consensus.judge_override")
    assert isinstance(judge_override, bool)
    assert judge_override is False


def test_judge_override_telemetry_flag(
    otel_exporter: InMemorySpanExporter,
) -> None:
    """Verify consensus_guard.consensus.judge_override is True when votes pass but verdict rejected."""
    # Simulate state where domain voters passed, but an explicit override / fatal flaw occurred
    mock_votes = [
        DomainVote(
            voter_name="schema_voter",
            domain="schema",
            score=0.90,
            confidence=0.9,
            passed=True,
            rationale="Schema clean.",
        ),
        DomainVote(
            voter_name="drift_voter",
            domain="drift",
            score=0.85,
            confidence=0.85,
            passed=True,
            rationale="No drift.",
        ),
    ]

    state: dict[str, Any] = {
        "votes": mock_votes,
        "judge_override": True,
        "overrides_applied": ["Fatal Flaw: Target Completeness < 95%"],
    }

    res = asyncio.run(consensus_reducer(state))
    assert res["judge_override"] is True

    spans = otel_exporter.get_finished_spans()
    reducer_spans = [s for s in spans if "consensus_reducer" in s.name]
    assert len(reducer_spans) == 1
    attrs = reducer_spans[0].attributes or {}

    assert attrs.get("consensus_guard.consensus.judge_override") is True
    assert isinstance(attrs.get("consensus_guard.consensus.voter_variance"), (int, float))


def test_direct_telemetry_functions(
    otel_exporter: InMemorySpanExporter,
) -> None:
    """Test direct invocation of layer1_gate, domain_voters_fanout, and consensus_reducer."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    state: dict[str, Any] = {"raw_data": df}

    # 1. layer1_gate
    gate_out = asyncio.run(layer1_gate(state))
    state.update(gate_out)

    # 2. domain_voters_fanout
    fanout_out = asyncio.run(domain_voters_fanout(state))
    state.update(fanout_out)

    # 3. consensus_reducer
    reducer_out = asyncio.run(consensus_reducer(state))
    state.update(reducer_out)

    spans = otel_exporter.get_finished_spans()
    assert len(spans) == 3

    names = {s.name for s in spans}
    assert "consensus_guard.layer1_gate" in names
    assert "consensus_guard.domain_voters_fanout" in names
    assert "consensus_guard.consensus_reducer" in names


def test_estimate_tokens_saved_calculation() -> None:
    """Verify estimate_tokens_saved produces positive scaling token values."""
    assert estimate_tokens_saved(None) > 0

    from consensus_guard.layer1 import Layer1Stats

    stats = Layer1Stats(
        row_count=500,
        column_count=10,
        null_ratios={},
        cardinality_ratios={},
        drift_slopes={},
        drift_sigmas={},
    )
    tokens = estimate_tokens_saved(stats, num_voters=2)
    # Base 5,500 + 500 (col overhead) + 50*10*3 (1,500 data overhead) = ~7,500
    assert tokens >= 7000
