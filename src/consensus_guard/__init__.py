"""consensus_guard: Core deterministic mathematical guardrails and statistical gates."""

from __future__ import annotations

from consensus_guard.graph import (
    ConsensusState,
    StandaloneConsensusGraphRunner,
    build_consensus_graph,
    consensus_reducer_node,
    drift_consensus_voter_node,
    layer1_gate_node,
    route_after_gate,
    schema_integrity_voter_node,
    short_circuit_rejection_node,
)
from consensus_guard.layer1 import (
    GateResult,
    Layer1Stats,
    compute_layer1_stats,
    evaluate_statistical_invariants,
)
from consensus_guard.voters import (
    DomainVote,
    drift_consensus_voter,
    evaluate_drift_consensus,
    evaluate_schema_integrity,
    schema_integrity_voter,
)

__all__ = [
    # Graph & Reducers
    "ConsensusState",
    # Domain Voters
    "DomainVote",
    # Layer 1 Gate
    "GateResult",
    "Layer1Stats",
    "StandaloneConsensusGraphRunner",
    "build_consensus_graph",
    "compute_layer1_stats",
    "consensus_reducer_node",
    "drift_consensus_voter",
    "drift_consensus_voter_node",
    "evaluate_drift_consensus",
    "evaluate_schema_integrity",
    "evaluate_statistical_invariants",
    "layer1_gate_node",
    "route_after_gate",
    "schema_integrity_voter",
    "schema_integrity_voter_node",
    "short_circuit_rejection_node",
]
