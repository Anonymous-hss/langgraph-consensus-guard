"""Core state schema models for LangGraph Consensus Guard.

Defines the mathematical Layer 1 metric results, Layer 2 domain agent votes,
Layer 3 consensus verdicts, and the LangGraph-compatible state container.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict


class Disposition(str, Enum):
    """Corpus or dataset disposition ordered by severity.

    Severity ranking:
    READY (0) < CONDITIONAL (1) < SUSPENDED (2).
    A downstream guard or agent may escalate severity, but never downgrade it.
    """

    READY = "ready"
    CONDITIONAL = "conditional"
    SUSPENDED = "suspended"

    @property
    def severity(self) -> int:
        """Return numeric severity ranking."""
        mapping = {
            Disposition.READY: 0,
            Disposition.CONDITIONAL: 1,
            Disposition.SUSPENDED: 2,
        }
        return mapping[self]

    def escalate(self, candidate: Disposition | str) -> Disposition:
        """Return the more severe of self and candidate."""
        cand_enum = candidate if isinstance(candidate, Disposition) else Disposition(candidate)
        return cand_enum if cand_enum.severity > self.severity else self


class VerdictStatus(str, Enum):
    """Final consensus verdict outcomes."""

    READY = "READY"
    CONDITIONALLY_READY = "CONDITIONALLY_READY"
    FAILED = "FAILED"


RiskLevel = Literal["Low", "Medium", "High"]
TableRole = Literal["primary", "feature", "reference"]
CategoryType = Literal["metadata", "quality", "governance", "fairness"]


class ActionableFix(BaseModel):
    """Structured remediation action proposed by an expert agent or consensus judge."""

    model_config = ConfigDict(extra="allow", frozen=False)

    type: CategoryType = Field(
        ...,
        description="Category classification: 'metadata', 'quality', 'governance', or 'fairness'.",
    )
    addresses_check: str = Field(
        ...,
        description="Exact metric check or attribute this fix remedies (e.g. 'null_rate:price').",
    )
    description: str = Field(
        ...,
        description="Detailed description of the issue, business impact, and corrective action.",
    )
    usecase_validation: str = Field(
        default="",
        description="Explanation of how this remediation preserves downstream use-case validity.",
    )
    impact_assessment: str = Field(
        default="",
        description="Data volume impact and assessment of whether any data loss is acceptable.",
    )
    auto_apply: bool = Field(
        default=False,
        description="Whether this fix is non-destructive and can be executed without manual sign-off.",
    )
    risk_level: RiskLevel = Field(
        default="Medium",
        description="Risk level: 'Low' (no loss), 'Medium' (partial change), 'High' (destructive).",
    )
    code: str = Field(
        default="",
        description="Executable Python or SQL transformation code modifying the dataset in-place.",
    )


class MetricCheck(BaseModel):
    """Individual atomic metric check result evaluated mathematically in Layer 1."""

    model_config = ConfigDict(extra="allow", frozen=False)

    name: str = Field(..., description="Unique check name, e.g. 'completeness_score'.")
    passed: bool = Field(..., description="Whether this specific check met the threshold.")
    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Normalized score between 0.0 and 10.0 for this individual check.",
    )
    value: Any = Field(default=None, description="Observed/computed mathematical value.")
    threshold: Any = Field(default=None, description="Threshold required for passing.")
    detail: str = Field(default="", description="Human-readable diagnostics or context.")
    failing_items: list[dict[str, Any]] | None = Field(
        default=None,
        description="Bounded sample of failing records/columns for failure attribution.",
    )


class MetricCategoryResult(BaseModel):
    """Aggregated results for an entire metric category (e.g. quality, governance)."""

    model_config = ConfigDict(extra="allow", frozen=False)

    category: str = Field(..., description="Category identifier, e.g. 'quality' or 'governance'.")
    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Category aggregate score normalized from 0.0 to 10.0.",
    )
    passed: bool = Field(..., description="True if all critical checks in this category passed.")
    checks: list[MetricCheck] = Field(
        default_factory=lambda: list[MetricCheck](),
        description="Individual atomic check results belonging to this category.",
    )
    failures: list[str] = Field(
        default_factory=lambda: list[str](),
        description="List of check names or diagnostic messages that failed.",
    )


class Layer1MetricResult(BaseModel):
    """Layer 1 deterministic mathematical metric evaluation.

    Contains pure arithmetic and rule-based calculations computed directly
    on the data artifacts prior to any LLM execution. Serves as the immutable
    ground truth to which LLM consensus is anchored.
    """

    model_config = ConfigDict(extra="allow", frozen=False)

    run_id: str = Field(..., description="Unique execution run identifier.")
    dataset_ref: str = Field(..., description="Table name, URI, or corpus identifier.")
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Deterministic baseline score (0.0 to 10.0) from Layer 1 engine.",
    )
    passed: bool = Field(..., description="Whether the baseline score meets passing threshold.")
    categories: dict[str, MetricCategoryResult] = Field(
        default_factory=lambda: dict[str, MetricCategoryResult](),
        description="Categorized evaluation results (quality, governance, schema, fairness).",
    )
    detected_pii: list[dict[str, Any]] = Field(
        default_factory=lambda: list[dict[str, Any]](),
        description="List of detected PII entities with column/text, entity type, and confidence.",
    )
    null_rates: dict[str, float] = Field(
        default_factory=lambda: dict[str, float](),
        description="Per-column null rate percentages [0.0 - 1.0].",
    )
    duplicate_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of exact duplicate rows to total rows.",
    )
    schema_complexity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Structural schema complexity measure (narrow/homogeneous schemas < 0.5).",
    )
    total_rows: int = Field(
        default=0,
        ge=0,
        description="Total row or chunk count evaluated.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=lambda: dict[str, Any](),
        description="Arbitrary provenance, data source metadata, or execution timestamps.",
    )

    def get_category_score(self, category: str, default: float = 0.0) -> float:
        """Retrieve the score for a specific category or return default."""
        cat = self.categories.get(category)
        return cat.score if cat is not None else default

    def has_critical_pii(self, critical_types: set[str] | None = None) -> bool:
        """Check if any critical unmasked PII (e.g. SSN, CREDIT_CARD, PASSPORT) is present."""
        targets = critical_types or {"SSN", "CREDIT_CARD", "PASSPORT"}
        target_lower = {t.lower() for t in targets}
        for item in self.detected_pii:
            entity_type = str(item.get("type", "")).strip().lower()
            col = str(item.get("column", "")).strip().lower()
            if entity_type in target_lower or col in target_lower:
                return True
        return False


class DomainAgentVote(BaseModel):
    """Layer 2 expert agent qualitative assessment and vote.

    Produced by specialized domain experts (e.g., Data Quality Analyst,
    Governance Guardian, Fairness Expert, Schema Architect, Business Alignment).
    """

    model_config = ConfigDict(extra="allow", frozen=False)

    agent_id: str = Field(..., description="Unique agent identifier, e.g. 'governance_guardian'.")
    domain: str = Field(
        ...,
        description="Subject domain, e.g. 'governance', 'quality', 'fairness', 'schema'.",
    )
    disposition: Disposition = Field(
        default=Disposition.READY,
        description="Agent disposition vote: 'ready', 'conditional', or 'suspended'.",
    )
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Domain score (0.0 to 10.0) assessed by this agent, if applicable.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent certainty in its evaluation.",
    )
    summary: str = Field(..., description="One to two sentence executive summary.")
    findings: list[str] = Field(
        default_factory=lambda: list[str](),
        description="Specific flagged issues, anomalies, or governance breaches.",
    )
    suggested_fixes: list[ActionableFix] = Field(
        default_factory=lambda: list[ActionableFix](),
        description="Recommended remediation fixes generated by this agent.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=lambda: dict[str, Any](),
        description="Agent-specific metrics, diagnostics, or reasoning traces.",
    )


class ConsensusVerdict(BaseModel):
    """Layer 3 consensus synthesis output.

    Combines deterministic mathematical ground truth from Layer 1 with
    qualitative assessments from Layer 2 domain agents, applying deterministic
    anchoring (±2.5 band), fatal flaw overrides, and disposition escalation.
    """

    model_config = ConfigDict(extra="allow", frozen=False)

    verdict: VerdictStatus = Field(
        ...,
        description="Final operational verdict: 'READY', 'CONDITIONALLY_READY', or 'FAILED'.",
    )
    global_readiness_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Consensus score mathematically anchored to Layer 1 baseline.",
    )
    deterministic_baseline_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Pristine unadjusted Layer 1 score from MetricEngine.",
    )
    disposition: Disposition = Field(
        ...,
        description="Final disposition: 'ready', 'conditional', or 'suspended'.",
    )
    category_scores: dict[str, float] = Field(
        default_factory=lambda: dict[str, float](),
        description="Calibrated score per domain category (quality, governance, schema, fairness).",
    )
    consensus_explanation: str = Field(
        ...,
        description="Synthesis narrative detailing rationale, trade-offs, and override explanations.",
    )
    overrides_applied: list[str] = Field(
        default_factory=lambda: list[str](),
        description="Deterministic override rules triggered (e.g. critical PII caps).",
    )
    suggestions: list[ActionableFix] = Field(
        default_factory=lambda: list[ActionableFix](),
        description="Ranked, deduplicated remediation fixes approved by the judge.",
    )
    agent_votes: dict[str, DomainAgentVote] = Field(
        default_factory=lambda: dict[str, DomainAgentVote](),
        description="Map of domain agent votes evaluated in consensus.",
    )
    passed: bool = Field(
        ...,
        description="Whether the dataset is cleared for production consumption.",
    )


def merge_votes(
    current: dict[str, DomainAgentVote] | None,
    new_votes: dict[str, DomainAgentVote] | list[DomainAgentVote] | None,
) -> dict[str, DomainAgentVote]:
    """LangGraph state reducer to merge expert agent votes into state."""
    result: dict[str, DomainAgentVote] = dict(current or {})
    if not new_votes:
        return result

    if isinstance(new_votes, list):
        for vote in new_votes:
            result[vote.agent_id] = vote
    else:
        result.update(new_votes)
    return result


class ConsensusGuardState(TypedDict):
    """Typed state dictionary for LangGraph StateGraph execution."""

    run_id: str
    dataset_ref: str
    usecase: str
    table_role: TableRole
    target_column: str | None
    iteration: int
    max_iterations: int
    passing_threshold: float

    # Layer 1 Metrics
    layer1_metrics: Layer1MetricResult | dict[str, Any] | None

    # Layer 2 Domain Agent Votes (merged via reducer)
    domain_votes: dict[str, DomainAgentVote]

    # Layer 3 Consensus Output
    consensus_verdict: ConsensusVerdict | None

    # Audit & Execution Streams
    agent_dialogue: list[dict[str, Any]]
    overrides_applied: list[str]
    suggestions: list[ActionableFix]
    applied_fixes_history: list[str]

    # Workflow lifecycle
    status: str
    error: str | None

    # Optional extension fields
    headless_mode: NotRequired[bool]
    approval_status: NotRequired[str]
