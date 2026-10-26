"""consensus_guard: Core deterministic mathematical guardrails and statistical gates."""

from __future__ import annotations

from consensus_guard.layer1 import (
    GateResult,
    Layer1Stats,
    compute_layer1_stats,
    evaluate_statistical_invariants,
)

__all__ = [
    "GateResult",
    "Layer1Stats",
    "compute_layer1_stats",
    "evaluate_statistical_invariants",
]
