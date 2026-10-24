"""LangGraph StateGraph workflow for multi-agent consensus guardrails.

Orchestrates the 3-layer guardrail pipeline:
- START -> evaluate_layer1 -> collect_domain_votes -> consensus_judge -> route_verdict
  -> [push_to_target | await_approval | end_failed]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from langgraph_consensus_guard.anchoring import AnchoringConfig, synthesize_consensus
from langgraph_consensus_guard.schema import (
    ConsensusGuardState,
    ConsensusVerdict,
    Disposition,
    DomainAgentVote,
    Layer1MetricResult,
    VerdictStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

RoutingDestination = Literal["push_to_target", "await_approval", "end_failed"]


# ---------------------------------------------------------------------------
# Node Implementations
# ---------------------------------------------------------------------------


async def evaluate_layer1_node(
    state: ConsensusGuardState,
    dataset_evaluator: Callable[[ConsensusGuardState], Layer1MetricResult] | None = None,
) -> dict[str, Any]:
    """Node 1: Compute deterministic Layer 1 metrics before any LLM execution."""
    logger.info("Executing Layer 1 deterministic metric evaluation for %s", state["run_id"])

    if dataset_evaluator is not None:
        metrics = dataset_evaluator(state)
    elif state.get("layer1_metrics") is not None:
        raw_m = state["layer1_metrics"]
        if isinstance(raw_m, Layer1MetricResult):
            metrics = raw_m
        elif isinstance(raw_m, dict):
            metrics = Layer1MetricResult.model_validate(raw_m)
        else:
            metrics = Layer1MetricResult(
                run_id=state["run_id"],
                dataset_ref=state["dataset_ref"],
                overall_score=5.0,
                passed=False,
            )
    else:
        # Default fallback metric computation
        metrics = Layer1MetricResult(
            run_id=state["run_id"],
            dataset_ref=state["dataset_ref"],
            overall_score=7.5,
            passed=True,
        )

    return {
        "layer1_metrics": metrics,
        "status": "evaluating_layer1_done",
    }


async def collect_domain_votes_node(
    state: ConsensusGuardState,
    vote_collectors: list[Callable[[ConsensusGuardState], DomainAgentVote]] | None = None,
) -> dict[str, Any]:
    """Node 2: Collect qualitative assessments from Layer 2 domain expert agents."""
    logger.info("Collecting Layer 2 domain agent votes for %s", state["run_id"])

    collected_votes: dict[str, DomainAgentVote] = dict(state.get("domain_votes") or {})

    if vote_collectors:
        for collector in vote_collectors:
            try:
                vote = collector(state)
                collected_votes[vote.agent_id] = vote
            except Exception as exc:
                logger.error("Domain vote collector failed: %s", exc)

    return {
        "domain_votes": collected_votes,
        "status": "domain_votes_collected",
    }


async def consensus_judge_node(
    state: ConsensusGuardState,
    config: AnchoringConfig | None = None,
) -> dict[str, Any]:
    """Node 3: Synthesize consensus verdict anchored to Layer 1 mathematical baseline."""
    logger.info("Synthesizing Layer 3 consensus verdict for %s", state["run_id"])

    cfg = config or AnchoringConfig(passing_threshold=state.get("passing_threshold", 8.0))

    raw_metrics = state.get("layer1_metrics")
    if isinstance(raw_metrics, Layer1MetricResult):
        layer1 = raw_metrics
    elif isinstance(raw_metrics, dict):
        layer1 = Layer1MetricResult.model_validate(raw_metrics)
    else:
        layer1 = Layer1MetricResult(
            run_id=state["run_id"],
            dataset_ref=state["dataset_ref"],
            overall_score=5.0,
            passed=False,
        )

    domain_votes = state.get("domain_votes") or {}
    table_role = state.get("table_role", "primary")
    target_col = state.get("target_column")
    fixes_count = len(state.get("applied_fixes_history") or [])

    verdict = synthesize_consensus(
        layer1_metrics=layer1,
        domain_votes=domain_votes,
        table_role=table_role,
        target_column=target_col,
        applied_fixes_count=fixes_count,
        config=cfg,
    )

    return {
        "consensus_verdict": verdict,
        "overrides_applied": verdict.overrides_applied,
        "suggestions": verdict.suggestions,
        "status": "consensus_verdict_reached",
    }


def route_consensus_verdict(state: ConsensusGuardState) -> RoutingDestination:
    """Routing function determining next step based on consensus verdict and disposition."""
    verdict: ConsensusVerdict | None = state.get("consensus_verdict")
    if verdict is None:
        return "end_failed"

    if verdict.verdict == VerdictStatus.READY and verdict.disposition == Disposition.READY:
        return "push_to_target"

    if (
        verdict.verdict == VerdictStatus.CONDITIONALLY_READY
        or verdict.disposition == Disposition.CONDITIONAL
    ):
        return "await_approval"

    return "end_failed"


async def push_to_target_node(state: ConsensusGuardState) -> dict[str, Any]:
    """Terminal Node: Dataset cleared by consensus guard for production pipeline."""
    logger.info("Dataset %s cleared and routed to target", state["dataset_ref"])
    return {"status": "cleared_push_to_target"}


async def await_approval_node(state: ConsensusGuardState) -> dict[str, Any]:
    """Pause Node: Dataset conditionally approved; awaiting operator review or fix sign-off."""
    logger.info("Dataset %s placed on conditional hold", state["dataset_ref"])
    return {"status": "awaiting_approval"}


async def end_failed_node(state: ConsensusGuardState) -> dict[str, Any]:
    """Terminal Node: Dataset suspended or fatal flaw encountered."""
    logger.warning("Dataset %s failed consensus guardrails", state["dataset_ref"])
    return {"status": "failed_consensus_guard"}


# ---------------------------------------------------------------------------
# Graph Builder & Runner
# ---------------------------------------------------------------------------


def build_consensus_guard_graph(
    anchoring_config: AnchoringConfig | None = None,
    dataset_evaluator: Callable[[ConsensusGuardState], Layer1MetricResult] | None = None,
    vote_collectors: list[Callable[[ConsensusGuardState], DomainAgentVote]] | None = None,
) -> Any:
    """Build and compile the LangGraph StateGraph.

    If the `langgraph` package is installed, returns a compiled LangGraph Runnable.
    If `langgraph` is not installed, returns a StandaloneConsensusGuardRunner that
    executes the identical pipeline structure and state transitions.
    """
    cfg = anchoring_config or AnchoringConfig()

    try:
        from langgraph.graph import END, START, StateGraph

        graph: StateGraph[ConsensusGuardState] = StateGraph(ConsensusGuardState)

        # Register nodes with curried callbacks
        async def _eval_l1(s: ConsensusGuardState) -> dict[str, Any]:
            return await evaluate_layer1_node(s, dataset_evaluator=dataset_evaluator)

        async def _collect_v(s: ConsensusGuardState) -> dict[str, Any]:
            return await collect_domain_votes_node(s, vote_collectors=vote_collectors)

        async def _judge(s: ConsensusGuardState) -> dict[str, Any]:
            return await consensus_judge_node(s, config=cfg)

        graph.add_node("evaluate_layer1", _eval_l1)
        graph.add_node("collect_domain_votes", _collect_v)
        graph.add_node("consensus_judge", _judge)
        graph.add_node("push_to_target", push_to_target_node)
        graph.add_node("await_approval", await_approval_node)
        graph.add_node("end_failed", end_failed_node)

        # Edges
        graph.add_edge(START, "evaluate_layer1")
        graph.add_edge("evaluate_layer1", "collect_domain_votes")
        graph.add_edge("collect_domain_votes", "consensus_judge")

        # Conditional routing after consensus judge
        graph.add_conditional_edges(
            "consensus_judge",
            route_consensus_verdict,
            {
                "push_to_target": "push_to_target",
                "await_approval": "await_approval",
                "end_failed": "end_failed",
            },
        )

        graph.add_edge("push_to_target", END)
        graph.add_edge("await_approval", END)
        graph.add_edge("end_failed", END)

        return graph.compile()

    except ImportError:
        logger.info("LangGraph not installed; using StandaloneConsensusGuardRunner")
        return StandaloneConsensusGuardRunner(
            config=cfg,
            dataset_evaluator=dataset_evaluator,
            vote_collectors=vote_collectors,
        )


class StandaloneConsensusGuardRunner:
    """Standalone runner providing LangGraph-compatible state workflow execution."""

    def __init__(
        self,
        config: AnchoringConfig | None = None,
        dataset_evaluator: Callable[[ConsensusGuardState], Layer1MetricResult] | None = None,
        vote_collectors: list[Callable[[ConsensusGuardState], DomainAgentVote]] | None = None,
    ) -> None:
        self.config = config or AnchoringConfig()
        self.dataset_evaluator = dataset_evaluator
        self.vote_collectors = vote_collectors

    async def ainvoke(self, state: ConsensusGuardState) -> ConsensusGuardState:
        """Asynchronously invoke the consensus guard workflow."""
        curr_state: ConsensusGuardState = dict(state)  # type: ignore[assignment]

        # 1. Layer 1 evaluation
        l1_res = await evaluate_layer1_node(curr_state, self.dataset_evaluator)
        curr_state.update(l1_res)  # type: ignore[arg-type]

        # 2. Collect domain votes
        votes_res = await collect_domain_votes_node(curr_state, self.vote_collectors)
        curr_state.update(votes_res)  # type: ignore[arg-type]

        # 3. Consensus Judge
        judge_res = await consensus_judge_node(curr_state, self.config)
        curr_state.update(judge_res)  # type: ignore[arg-type]

        # 4. Route
        route = route_consensus_verdict(curr_state)
        if route == "push_to_target":
            term_res = await push_to_target_node(curr_state)
        elif route == "await_approval":
            term_res = await await_approval_node(curr_state)
        else:
            term_res = await end_failed_node(curr_state)

        curr_state.update(term_res)  # type: ignore[arg-type]
        return curr_state

    def invoke(self, state: ConsensusGuardState) -> ConsensusGuardState:
        """Synchronously execute the consensus guard workflow."""
        import asyncio

        return asyncio.run(self.ainvoke(state))
