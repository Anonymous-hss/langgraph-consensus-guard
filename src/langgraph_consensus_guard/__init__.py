"""langgraph-consensus-guard: Multi-agent consensus guardrails with deterministic mathematical anchoring.

Exports core state schemas (Layer1MetricResult, DomainAgentVote, ConsensusVerdict),
anchoring algorithms, fatal flaw overrides, and the ConsensusGuardGraph workflow.
"""

from __future__ import annotations

from langgraph_consensus_guard.anchoring import (
    AnchoringConfig,
    anchor_score,
    apply_fatal_flaw_overrides,
    escalate_disposition,
    graded_cap,
    synthesize_consensus,
)
from langgraph_consensus_guard.engine import (
    CRITICAL_PII_TYPES,
    compute_shannon_entropy,
    evaluate_records,
)
from langgraph_consensus_guard.graph import (
    StandaloneConsensusGuardRunner,
    await_approval_node,
    build_consensus_guard_graph,
    collect_domain_votes_node,
    consensus_judge_node,
    end_failed_node,
    evaluate_layer1_node,
    push_to_target_node,
    route_consensus_verdict,
)
from langgraph_consensus_guard.schema import (
    ActionableFix,
    CategoryType,
    ConsensusGuardState,
    ConsensusVerdict,
    Disposition,
    DomainAgentVote,
    Layer1MetricResult,
    MetricCategoryResult,
    MetricCheck,
    RiskLevel,
    TableRole,
    VerdictStatus,
    merge_votes,
)

__version__ = "0.1.0"

__all__ = [
    "CRITICAL_PII_TYPES",
    "ActionableFix",
    # Anchoring & Rules
    "AnchoringConfig",
    "CategoryType",
    "ConsensusGuardState",
    "ConsensusVerdict",
    "Disposition",
    "DomainAgentVote",
    # Schemas
    "Layer1MetricResult",
    "MetricCategoryResult",
    "MetricCheck",
    "RiskLevel",
    "StandaloneConsensusGuardRunner",
    "TableRole",
    "VerdictStatus",
    "anchor_score",
    "apply_fatal_flaw_overrides",
    "await_approval_node",
    # Graph & Nodes
    "build_consensus_guard_graph",
    "collect_domain_votes_node",
    "compute_shannon_entropy",
    "consensus_judge_node",
    "end_failed_node",
    "escalate_disposition",
    "evaluate_layer1_node",
    # Deterministic Engine
    "evaluate_records",
    "graded_cap",
    "merge_votes",
    "push_to_target_node",
    "route_consensus_verdict",
    "synthesize_consensus",
]
