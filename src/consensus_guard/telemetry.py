"""OpenTelemetry instrumentation for Consensus Guard.

Instruments:
- layer1_gate: Deterministic statistical validation, short-circuit detection, and tokens saved estimation.
- domain_voters_fanout: Concurrent domain expert voter fan-out.
- consensus_reducer: Vote aggregation, variance analysis, and judge override detection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import opentelemetry.trace as trace

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

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

TRACER_NAME = "langgraph.consensus_guard"


def get_tracer() -> trace.Tracer:
    """Return the Consensus Guard OpenTelemetry tracer."""
    return trace.get_tracer(TRACER_NAME)


def estimate_tokens_saved(
    stats: Layer1Stats | None,
    num_voters: int = 2,
    base_voter_tokens: int = 1500,
    judge_tokens: int = 2500,
) -> int:
    """Estimate the number of LLM tokens saved by short-circuiting at Layer 1.

    When Layer 1 invariants fail, downstream Layer 2 Domain Expert Agent LLM calls
    and the Layer 3 Consensus Judge LLM call are bypassed entirely.

    Calculation:
    - Base prompt + completion budget for each bypassed domain agent (~1,500 tokens).
    - Base prompt + reasoning/completion budget for Consensus Judge (~2,500 tokens).
    - Payload serialization overhead scaled by schema columns and sample rows.
    """
    base_tokens = (num_voters * base_voter_tokens) + judge_tokens
    if stats is None:
        return int(base_tokens)

    # Schema definition and metadata prompt tokens (~50 tokens per column)
    col_overhead = stats.column_count * 50
    # Data row serialization tokens (~3 tokens per cell on sample rows sent to prompt)
    sample_rows = min(stats.row_count, 50)
    data_overhead = sample_rows * max(1, stats.column_count) * 3

    return int(base_tokens + col_overhead + data_overhead)


async def layer1_gate(state: Any) -> dict[str, Any]:
    """Execute Layer 1 Gate wrapped with OpenTelemetry instrumentation."""
    tracer = get_tracer()
    with tracer.start_as_current_span("consensus_guard.layer1_gate") as span:
        raw_data = state.get("raw_data") if hasattr(state, "get") else None
        stats = state.get("stats") if hasattr(state, "get") else None

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

        meta = state.get("metadata") if hasattr(state, "get") else {}
        meta_dict = meta if isinstance(meta, dict) else {}
        max_null = float(meta_dict.get("max_null_ratio", 0.05))
        max_drift = float(meta_dict.get("max_drift_sigma", 3.0))

        gate_result: GateResult = evaluate_statistical_invariants(
            stats=stats,
            max_null_ratio=max_null,
            max_drift_sigma=max_drift,
        )

        short_circuited = bool(gate_result.short_circuit or not gate_result.passed)
        obs_max_null = (
            float(max(stats.null_ratios.values()))
            if stats.null_ratios
            else 0.0
        )
        obs_max_drift = (
            float(max(stats.drift_sigmas.values()))
            if stats.drift_sigmas
            else 0.0
        )

        tokens_saved = estimate_tokens_saved(stats) if short_circuited else 0

        # Explicit span attributes required by specification
        span.set_attribute("consensus_guard.layer1.short_circuited", short_circuited)
        span.set_attribute("consensus_guard.layer1.max_null_ratio", obs_max_null)
        span.set_attribute("consensus_guard.layer1.max_ols_drift_sigma", obs_max_drift)
        span.set_attribute("consensus_guard.tokens_saved_estimate", tokens_saved)

        return {
            "stats": stats,
            "gate_result": gate_result,
            "tokens_saved_estimate": tokens_saved,
        }


async def domain_voters_fanout(
    state: Any,
    voter_nodes: list[Callable[[Any], Any]] | None = None,
) -> dict[str, Any]:
    """Execute concurrent domain expert voter fan-out wrapped with OpenTelemetry."""
    tracer = get_tracer()
    with tracer.start_as_current_span("consensus_guard.domain_voters_fanout") as span:
        if voter_nodes is None:
            voter_nodes = [schema_integrity_voter, drift_consensus_voter]

        # Handle both sync and async voter nodes seamlessly
        voter_coros: list[Any] = []
        for node in voter_nodes:
            res = node(dict(state) if hasattr(state, "items") else state)
            if asyncio.iscoroutine(res):
                voter_coros.append(res)
            else:
                async def _wrap(sync_res: Any = res) -> Any:
                    return sync_res
                voter_coros.append(_wrap())

        voter_outputs = await asyncio.gather(*voter_coros)

        reduced_votes: list[DomainVote] = list(state.get("votes") or []) if hasattr(state, "get") else []
        for out in voter_outputs:
            if isinstance(out, dict) and "votes" in out and isinstance(out["votes"], list):
                reduced_votes.extend(out["votes"])

        span.set_attribute("consensus_guard.voter_count", len(voter_nodes))
        span.set_attribute("consensus_guard.votes_collected", len(reduced_votes))

        return {"votes": reduced_votes}


async def consensus_reducer(state: Any) -> dict[str, Any]:
    """Execute consensus reducer wrapped with OpenTelemetry."""
    tracer = get_tracer()
    with tracer.start_as_current_span("consensus_guard.consensus_reducer") as span:
        votes: list[DomainVote] = state.get("votes") or [] if hasattr(state, "get") else []
        if not votes:
            res: dict[str, Any] = {
                "consensus_score": 0.0,
                "consensus_verdict": "REJECTED",
                "short_circuited": False,
                "rejection_reason": "No domain votes recorded by consensus reducer.",
            }
            span.set_attribute("consensus_guard.consensus.voter_variance", 0.0)
            span.set_attribute("consensus_guard.consensus.judge_override", False)
            span.set_attribute("consensus_guard.consensus.score", 0.0)
            span.set_attribute("consensus_guard.consensus.verdict", "REJECTED")
            return res

        scores = [float(v.score) for v in votes]
        voter_variance = float(np.var(scores)) if len(scores) > 1 else 0.0

        total_weight = sum(max(0.1, v.confidence) for v in votes)
        weighted_score = sum(v.score * max(0.1, v.confidence) for v in votes) / total_weight
        all_passed = all(v.passed for v in votes)

        verdict = "APPROVED" if (all_passed and weighted_score >= 0.70) else "REJECTED"

        # Check for judge override
        voter_consensus_pass = all_passed
        judge_override = bool(
            (hasattr(state, "get") and state.get("judge_override"))
            or (hasattr(state, "get") and state.get("overrides_applied"))
            or (voter_consensus_pass != (verdict == "APPROVED"))
        )

        span.set_attribute("consensus_guard.consensus.voter_variance", round(voter_variance, 6))
        span.set_attribute("consensus_guard.consensus.judge_override", judge_override)
        span.set_attribute("consensus_guard.consensus.score", round(float(weighted_score), 4))
        span.set_attribute("consensus_guard.consensus.verdict", verdict)

        return {
            "consensus_score": round(float(weighted_score), 4),
            "consensus_verdict": verdict,
            "short_circuited": False,
            "rejection_reason": None if verdict == "APPROVED" else "Domain vote consensus threshold not satisfied.",
            "voter_variance": round(voter_variance, 6),
            "judge_override": judge_override,
        }
