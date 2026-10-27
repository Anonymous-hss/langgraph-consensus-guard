"""LangGraph StateGraph for Consensus Guard concurrent domain fan-out.

Constructs the routing topology:
Input -> Layer 1 Gate -> [Conditional Edge: Pass -> Fan-out to Voters | Fail -> Short-Circuit Rejection]
      -> Consensus Reducer -> Output.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from consensus_guard.layer1 import (
    GateResult,
    Layer1Stats,
    compute_layer1_stats,
    evaluate_statistical_invariants,
)
from consensus_guard.voters import (
    DomainVote,
    drift_consensus_voter,
    schema_integrity_voter,
)

logger = logging.getLogger(__name__)


class ConsensusState(TypedDict):
    """Workflow state dictionary with Pydantic reducers for LangGraph execution."""

    raw_data: Any
    stats: Layer1Stats | None
    gate_result: GateResult | None
    votes: Annotated[list[DomainVote], operator.add]
    consensus_score: float | None
    consensus_verdict: str | None
    short_circuited: bool
    rejection_reason: str | None
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Node Implementations
# ---------------------------------------------------------------------------


async def layer1_gate_node(state: ConsensusState) -> dict[str, Any]:
    """Node 1: Compute Layer 1 stats and evaluate deterministic statistical invariants."""
    raw_data = state.get("raw_data")
    stats = state.get("stats")

    if stats is None and raw_data is not None:
        stats = compute_layer1_stats(raw_data)

    if stats is None:
        stats = Layer1Stats(
            row_count=0,
            column_count=0,
            null_ratios={},
            cardinality_ratios={},
            drift_slopes={},
            drift_sigmas={},
        )

    meta = state.get("metadata") or {}
    max_null = float(meta.get("max_null_ratio", 0.05))
    max_drift = float(meta.get("max_drift_sigma", 3.0))

    gate_result = evaluate_statistical_invariants(
        stats=stats,
        max_null_ratio=max_null,
        max_drift_sigma=max_drift,
    )

    return {
        "stats": stats,
        "gate_result": gate_result,
    }


def route_after_gate(state: ConsensusState) -> list[str] | str:
    """Conditional routing edge: If invariant breached, short-circuit; else fan-out to voters."""
    gate = state.get("gate_result")
    if gate is None or gate.short_circuit or not gate.passed:
        return "short_circuit_rejection"

    # Concurrent fan-out to all domain voters
    return ["schema_integrity_voter", "drift_consensus_voter"]


async def short_circuit_rejection_node(state: ConsensusState) -> dict[str, Any]:
    """Short-circuit terminal node: abort execution without invoking any domain voter or LLM."""
    gate = state.get("gate_result")
    reason = gate.reason if gate and gate.reason else "Statistical Invariant Breach"
    logger.warning("Pipeline short-circuited at Layer 1: %s", reason)

    return {
        "short_circuited": True,
        "rejection_reason": reason,
        "consensus_verdict": "REJECTED",
        "consensus_score": 0.0,
    }


async def schema_integrity_voter_node(state: ConsensusState) -> dict[str, Any]:
    """Fan-out node: execute schema integrity analysis."""
    logger.debug("Executing concurrent voter: schema_integrity_voter")
    return schema_integrity_voter(dict(state))


async def drift_consensus_voter_node(state: ConsensusState) -> dict[str, Any]:
    """Fan-out node: execute feature drift consensus analysis."""
    logger.debug("Executing concurrent voter: drift_consensus_voter")
    return drift_consensus_voter(dict(state))


async def consensus_reducer_node(state: ConsensusState) -> dict[str, Any]:
    """Fan-in node: aggregate all Pydantic domain votes and issue final consensus verdict."""
    votes: list[DomainVote] = state.get("votes") or []
    if not votes:
        return {
            "consensus_score": 0.0,
            "consensus_verdict": "REJECTED",
            "short_circuited": False,
            "rejection_reason": "No domain votes recorded by consensus reducer.",
        }

    total_weight = sum(max(0.1, v.confidence) for v in votes)
    weighted_score = sum(v.score * max(0.1, v.confidence) for v in votes) / total_weight
    all_passed = all(v.passed for v in votes)

    verdict = "APPROVED" if (all_passed and weighted_score >= 0.70) else "REJECTED"

    return {
        "consensus_score": round(float(weighted_score), 4),
        "consensus_verdict": verdict,
        "short_circuited": False,
        "rejection_reason": None if verdict == "APPROVED" else "Domain vote consensus threshold not satisfied.",
    }


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------


def build_consensus_graph() -> Any:
    """Build the Consensus Guard LangGraph StateGraph with concurrent fan-out and Pydantic reducers.

    Topology:
        START -> layer1_gate -> [conditional: fail -> short_circuit_rejection | pass -> (voters)]
             -> schema_integrity_voter \
                                        -> consensus_reducer -> END
             -> drift_consensus_voter  /
        short_circuit_rejection -> END
    """
    try:
        from langgraph.graph import END, START, StateGraph

        graph: StateGraph[ConsensusState] = StateGraph(ConsensusState)

        graph.add_node("layer1_gate", layer1_gate_node)
        graph.add_node("short_circuit_rejection", short_circuit_rejection_node)
        graph.add_node("schema_integrity_voter", schema_integrity_voter_node)
        graph.add_node("drift_consensus_voter", drift_consensus_voter_node)
        graph.add_node("consensus_reducer", consensus_reducer_node)

        graph.add_edge(START, "layer1_gate")

        graph.add_conditional_edges(
            "layer1_gate",
            route_after_gate,
            {
                "short_circuit_rejection": "short_circuit_rejection",
                "schema_integrity_voter": "schema_integrity_voter",
                "drift_consensus_voter": "drift_consensus_voter",
            },
        )

        graph.add_edge("schema_integrity_voter", "consensus_reducer")
        graph.add_edge("drift_consensus_voter", "consensus_reducer")
        graph.add_edge("short_circuit_rejection", END)
        graph.add_edge("consensus_reducer", END)

        return graph.compile()

    except ImportError:
        logger.info("LangGraph not installed; utilizing StandaloneConsensusGraphRunner")
        return StandaloneConsensusGraphRunner()


class StandaloneConsensusGraphRunner:
    """Standalone runner implementing the exact StateGraph routing and concurrent fan-out."""

    async def ainvoke(self, state: ConsensusState | dict[str, Any]) -> ConsensusState:
        """Asynchronously execute workflow with true concurrent fan-out and Pydantic reduction."""
        curr: ConsensusState = {
            "raw_data": state.get("raw_data"),
            "stats": state.get("stats"),
            "gate_result": state.get("gate_result"),
            "votes": list(state.get("votes") or []),
            "consensus_score": state.get("consensus_score"),
            "consensus_verdict": state.get("consensus_verdict"),
            "short_circuited": state.get("short_circuited", False),
            "rejection_reason": state.get("rejection_reason"),
            "metadata": dict(state.get("metadata") or {}),
        }

        # 1. Layer 1 Gate Node
        gate_update = await layer1_gate_node(curr)
        curr.update(gate_update)  # type: ignore[arg-type]

        # 2. Conditional Routing
        route = route_after_gate(curr)
        if route == "short_circuit_rejection":
            rejection_update = await short_circuit_rejection_node(curr)
            curr.update(rejection_update)  # type: ignore[arg-type]
            return curr

        # 3. Concurrent Domain Fan-Out via asyncio.gather
        voter_coros = [
            schema_integrity_voter_node(curr),
            drift_consensus_voter_node(curr),
        ]
        voter_outputs = await asyncio.gather(*voter_coros)

        # 4. Pydantic Reducer (operator.add on votes list)
        reduced_votes: list[DomainVote] = list(curr.get("votes") or [])
        for out in voter_outputs:
            if "votes" in out and isinstance(out["votes"], list):
                reduced_votes.extend(out["votes"])
        curr["votes"] = reduced_votes

        # 5. Consensus Reducer Node
        reducer_update = await consensus_reducer_node(curr)
        curr.update(reducer_update)  # type: ignore[arg-type]

        return curr

    def invoke(self, state: ConsensusState | dict[str, Any]) -> ConsensusState:
        """Synchronously execute workflow."""
        return asyncio.run(self.ainvoke(state))
