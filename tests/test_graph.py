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
