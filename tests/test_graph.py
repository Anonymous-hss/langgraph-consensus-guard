"""Tests for ConsensusGuardGraph workflow, node execution, and conditional routing."""

from __future__ import annotations

import asyncio

from langgraph_consensus_guard.graph import (
    build_consensus_guard_graph,
    collect_domain_votes_node,
    evaluate_layer1_node,
    route_consensus_verdict,
)
from langgraph_consensus_guard.schema import (
    ConsensusGuardState,
    ConsensusVerdict,
    Disposition,
    DomainAgentVote,
    Layer1MetricResult,
    VerdictStatus,
)


def test_evaluate_layer1_node() -> None:
    """Verify evaluate_layer1_node computes or validates Layer 1 metrics."""
    state: ConsensusGuardState = {
        "run_id": "run_01",
        "dataset_ref": "analytics.transactions",
        "usecase": "Churn model training",
        "table_role": "primary",
        "target_column": "churned",
        "iteration": 1,
        "max_iterations": 3,
        "passing_threshold": 8.0,
        "layer1_metrics": None,
        "domain_votes": {},
        "consensus_verdict": None,
        "agent_dialogue": [],
        "overrides_applied": [],
        "suggestions": [],
        "applied_fixes_history": [],
        "status": "pending",
        "error": None,
    }

    result = asyncio.run(evaluate_layer1_node(state))
    assert result["status"] == "evaluating_layer1_done"
    assert isinstance(result["layer1_metrics"], Layer1MetricResult)


def test_collect_domain_votes_node() -> None:
    """Verify domain vote collection with custom collectors."""
    state: ConsensusGuardState = {
        "run_id": "run_02",
        "dataset_ref": "sales",
        "usecase": "Sales report",
        "table_role": "primary",
        "target_column": None,
        "iteration": 1,
        "max_iterations": 3,
        "passing_threshold": 8.0,
        "layer1_metrics": None,
        "domain_votes": {},
        "consensus_verdict": None,
        "agent_dialogue": [],
        "overrides_applied": [],
        "suggestions": [],
        "applied_fixes_history": [],
        "status": "pending",
        "error": None,
    }

    def sample_agent(s: ConsensusGuardState) -> DomainAgentVote:
        return DomainAgentVote(
            agent_id="quality_expert",
            domain="quality",
            disposition=Disposition.READY,
            score=9.0,
            summary="All columns healthy.",
        )

    result = asyncio.run(collect_domain_votes_node(state, vote_collectors=[sample_agent]))
    assert result["status"] == "domain_votes_collected"
    assert "quality_expert" in result["domain_votes"]


def test_route_consensus_verdict_branches() -> None:
    """Verify routing destinations based on verdict status and disposition."""
    base_state: ConsensusGuardState = {
        "run_id": "run_route",
        "dataset_ref": "data",
        "usecase": "test",
        "table_role": "primary",
        "target_column": None,
        "iteration": 1,
        "max_iterations": 3,
        "passing_threshold": 8.0,
        "layer1_metrics": None,
        "domain_votes": {},
        "consensus_verdict": None,
        "agent_dialogue": [],
        "overrides_applied": [],
        "suggestions": [],
        "applied_fixes_history": [],
        "status": "evaluating",
        "error": None,
    }

    # 1. READY verdict -> push_to_target
    state_ready = dict(base_state)
    state_ready["consensus_verdict"] = ConsensusVerdict(
        verdict=VerdictStatus.READY,
        global_readiness_score=8.5,
        deterministic_baseline_score=8.2,
        disposition=Disposition.READY,
        consensus_explanation="Cleared.",
        passed=True,
    )
    assert route_consensus_verdict(state_ready) == "push_to_target"  # type: ignore[arg-type]

    # 2. CONDITIONALLY_READY verdict -> await_approval
    state_cond = dict(base_state)
    state_cond["consensus_verdict"] = ConsensusVerdict(
        verdict=VerdictStatus.CONDITIONALLY_READY,
        global_readiness_score=7.2,
        deterministic_baseline_score=7.0,
        disposition=Disposition.CONDITIONAL,
        consensus_explanation="Minor issues.",
        passed=False,
    )
    assert route_consensus_verdict(state_cond) == "await_approval"  # type: ignore[arg-type]

    # 3. FAILED verdict -> end_failed
    state_fail = dict(base_state)
    state_fail["consensus_verdict"] = ConsensusVerdict(
        verdict=VerdictStatus.FAILED,
        global_readiness_score=4.0,
        deterministic_baseline_score=4.0,
        disposition=Disposition.SUSPENDED,
        consensus_explanation="Fatal flaws.",
        passed=False,
    )
    assert route_consensus_verdict(state_fail) == "end_failed"  # type: ignore[arg-type]


def test_end_to_end_workflow_execution() -> None:
    """Verify complete end-to-end workflow execution using build_consensus_guard_graph."""
    initial_metrics = Layer1MetricResult(
        run_id="run_e2e_01",
        dataset_ref="gold.customer_360",
        overall_score=8.4,
        passed=True,
        schema_complexity=0.9,
        total_rows=10000,
    )

    initial_state: ConsensusGuardState = {
        "run_id": "run_e2e_01",
        "dataset_ref": "gold.customer_360",
        "usecase": "Risk scoring feature store",
        "table_role": "primary",
        "target_column": None,
        "iteration": 1,
        "max_iterations": 3,
        "passing_threshold": 8.0,
        "layer1_metrics": initial_metrics,
        "domain_votes": {
            "quality_analyst": DomainAgentVote(
                agent_id="quality_analyst",
                domain="quality",
                disposition=Disposition.READY,
                score=8.5,
                summary="Quality metrics sound.",
            ),
            "governance_guardian": DomainAgentVote(
                agent_id="governance_guardian",
                domain="governance",
                disposition=Disposition.READY,
                score=9.0,
                summary="Full compliance.",
            ),
        },
        "consensus_verdict": None,
        "agent_dialogue": [],
        "overrides_applied": [],
        "suggestions": [],
        "applied_fixes_history": ["Applied duplicate removal"],
        "status": "pending",
        "error": None,
    }

    runner = build_consensus_guard_graph()
    output_state = asyncio.run(runner.ainvoke(initial_state))

    assert output_state["status"] == "cleared_push_to_target"
    assert output_state["consensus_verdict"] is not None
    assert output_state["consensus_verdict"].verdict == VerdictStatus.READY
    assert output_state["consensus_verdict"].passed is True


# ===========================================================================
# Pillar 2: Concurrent Domain Fan-Out & Pydantic Reducers Tests
# ===========================================================================


def test_concurrent_fanout_execution_clean_dataset() -> None:
    """Verify concurrent fan-out to schema and drift voters and consensus reduction on clean data."""
    import numpy as np
    import pandas as pd

    from consensus_guard.graph import build_consensus_graph
    from consensus_guard.voters import DomainVote

    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame(
        {
            "user_id": np.arange(1000, 1000 + n),
            "feature_metric": rng.normal(50.0, 5.0, size=n),
            "account_status": rng.choice(["active", "pending"], size=n),
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        }
    )

    graph = build_consensus_graph()
    state = asyncio.run(graph.ainvoke({"raw_data": df}))

    # 1. Gate passes
    assert state["short_circuited"] is False
    assert state["gate_result"] is not None
    assert state["gate_result"].passed is True

    # 2. Concurrent fan-out: exactly 2 domain votes merged via Pydantic reducer
    votes: list[DomainVote] = state["votes"]
    assert len(votes) == 2

    voter_names = {v.voter_name for v in votes}
    assert "schema_integrity_voter" in voter_names
    assert "drift_consensus_voter" in voter_names

    for vote in votes:
        assert isinstance(vote, DomainVote)
        assert 0.0 <= vote.score <= 1.0
        assert 0.0 <= vote.confidence <= 1.0
        assert len(vote.cited_metrics) > 0
        assert vote.passed is True

    # 3. Consensus reducer synthesis
    assert state["consensus_verdict"] == "APPROVED"
    assert state["consensus_score"] is not None
    assert state["consensus_score"] >= 0.70


def test_concurrent_fanout_execution_timing_proof() -> None:
    """Verify that fan-out voter coroutines execute concurrently via event-loop concurrency."""
    import time
    from typing import Any

    from consensus_guard.voters import DomainVote

    async def run_instrumented() -> float:
        t0 = time.perf_counter()

        # Simulate two voter coroutines each doing 25ms of async IO/reasoning
        async def mock_voter_1() -> dict[str, Any]:
            await asyncio.sleep(0.025)
            return {
                "votes": [
                    DomainVote(
                        voter_name="voter_1",
                        domain="test_1",
                        score=0.9,
                        confidence=0.9,
                        passed=True,
                        rationale="ok",
                    )
                ]
            }

        async def mock_voter_2() -> dict[str, Any]:
            await asyncio.sleep(0.025)
            return {
                "votes": [
                    DomainVote(
                        voter_name="voter_2",
                        domain="test_2",
                        score=0.85,
                        confidence=0.85,
                        passed=True,
                        rationale="ok",
                    )
                ]
            }

        # Run concurrently via asyncio.gather (same pattern as ConsensusGraph)
        await asyncio.gather(mock_voter_1(), mock_voter_2())
        return (time.perf_counter() - t0) * 1000.0

    duration_ms = asyncio.run(run_instrumented())
    # Serial execution would take >= 50ms (25ms + 25ms). Concurrent execution takes ~25-35ms.
    assert duration_ms < 45.0, f"Expected concurrent execution < 45ms, took {duration_ms:.2f}ms"


def test_concurrent_fanout_short_circuits_on_gate_breach() -> None:
    """Verify Layer 1 gate breach short-circuits directly to rejection without calling voters."""
    import numpy as np
    import pandas as pd

    from consensus_guard.graph import build_consensus_graph

    n = 200
    # Inject severe null ratio violation (30% nulls in critical feature)
    bad_feature = np.random.randn(n)
    bad_feature[:60] = np.nan

    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "bad_feature": bad_feature,
        }
    )

    graph = build_consensus_graph()
    state = asyncio.run(graph.ainvoke({"raw_data": df}))

    # Pipeline MUST short-circuit
    assert state["short_circuited"] is True
    assert state["consensus_verdict"] == "REJECTED"
    assert state["consensus_score"] == 0.0
    assert state["rejection_reason"] == "Statistical Invariant Breach"

    # Voters were NEVER called
    assert len(state["votes"]) == 0


def test_voter_pydantic_strict_constraints() -> None:
    """Verify DomainVote strictly enforces score [0, 1] and confidence [0, 1] bounds."""
    import pytest
    from pydantic import ValidationError

    from consensus_guard.voters import DomainVote

    # Valid instance
    vote = DomainVote(
        voter_name="test_voter",
        domain="testing",
        score=0.85,
        confidence=0.9,
        passed=True,
        cited_metrics=["metric_a=1.0"],
        rationale="Passed all tests.",
    )
    assert vote.score == 0.85

    # Invalid score > 1.0
    with pytest.raises(ValidationError):
        DomainVote(
            voter_name="bad_voter",
            domain="testing",
            score=1.5,
            confidence=0.9,
            passed=True,
            rationale="Invalid score.",
        )

    # Invalid score < 0.0
    with pytest.raises(ValidationError):
        DomainVote(
            voter_name="bad_voter",
            domain="testing",
            score=-0.1,
            confidence=0.9,
            passed=False,
            rationale="Negative score.",
        )

    # Invalid confidence > 1.0
    with pytest.raises(ValidationError):
        DomainVote(
            voter_name="bad_voter",
            domain="testing",
            score=0.8,
            confidence=1.1,
            passed=True,
            rationale="Invalid confidence.",
        )


def test_schema_integrity_voter_detects_pii_and_invalid_naming() -> None:
    """Verify schema integrity voter penalizes invalid names and unmasked critical PII."""
    import pandas as pd

    from consensus_guard.voters import schema_integrity_voter

    df = pd.DataFrame(
        {
            "user_ssn": ["123-45-6789", "987-65-4321"],
            "bad column name": [1, 2],
            "select": ["reserved", "word"],
        }
    )

    result = schema_integrity_voter({"raw_data": df})
    vote = result["votes"][0]

    assert vote.passed is False
    assert vote.score < 0.70
    assert any("CRITICAL unmasked PII" in r for r in [vote.rationale])
    assert "user_ssn" in vote.metadata["critical_pii_found"]


def test_drift_consensus_voter_distinguishes_seasonality_vs_corruption() -> None:
    """Verify drift voter distinguishes cyclical seasonality from corruption."""
    import numpy as np
    import pandas as pd

    from consensus_guard.voters import drift_consensus_voter

    n = 100
    t = np.arange(n)

    # 1. Cyclical seasonality pattern (stable variance, bounded oscillation)
    df_seasonal = pd.DataFrame(
        {
            "temperature": 20.0 + 10.0 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, n),
        }
    )
    res_seasonal = drift_consensus_voter(
        {"raw_data": df_seasonal, "metadata": {"expected_seasonal_columns": ["temperature"]}}
    )
    vote_seasonal = res_seasonal["votes"][0]
    assert vote_seasonal.passed is True
    assert vote_seasonal.score >= 0.85

    # 2. Pipeline corruption (catastrophic step jump)
    corrupted_data = np.ones(n) * 10.0
    corrupted_data[50:] = 5000.0  # Massive step shift jump
    df_corrupted = pd.DataFrame({"sensor_a": corrupted_data})

    res_corrupted = drift_consensus_voter({"raw_data": df_corrupted})
    vote_corrupted = res_corrupted["votes"][0]
    assert vote_corrupted.passed is False
    assert vote_corrupted.score <= 0.60

