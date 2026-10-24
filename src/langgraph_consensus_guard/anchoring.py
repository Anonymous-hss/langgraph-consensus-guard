"""Deterministic mathematical anchoring and fatal flaw override rules.

Implements the Layer 3 Consensus Judge guardrail engine:
- Anchors LLM or agent scores within a strict mathematical band (default ±2.5).
- Prevents optimistic LLM hallucinations from overriding mathematical failure.
- Applies deterministic fatal flaw caps (critical PII, target completeness, schema simplicity).
- Enforces severity-based disposition escalation (READY < CONDITIONAL < SUSPENDED).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langgraph_consensus_guard.schema import (
    ActionableFix,
    ConsensusVerdict,
    Disposition,
    DomainAgentVote,
    Layer1MetricResult,
    TableRole,
    VerdictStatus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchoringConfig:
    """Configuration parameters for mathematical score anchoring and fatal flaw caps."""

    anchor_band: float = 2.5
    passing_threshold: float = 8.0
    conditional_threshold: float = 6.0
    critical_pii_score_cap: float = 2.0
    target_null_score_cap: float = 3.0
    target_null_max_rate: float = 0.05
    schema_complexity_threshold: float = 0.5
    schema_complexity_score_cap: float = 7.9
    low_confidence_score_cap: float = 5.0
    min_business_confidence: float = 0.5
    min_rows_ml: int = 100
    low_rows_score_cap: float = 4.0
    noop_relaxation_max_lift: float = 0.5
    critical_pii_hints: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"ssn", "credit_card", "passport", "customer_ssn", "bank_account"}
        )
    )


def anchor_score(
    proposed_score: float,
    baseline_score: float,
    band: float = 2.5,
    asymmetric: bool = False,
) -> float:
    """Anchor an agent or LLM proposed score to the deterministic Layer 1 baseline.

    Parameters:
        proposed_score: The qualitative score proposed by an agent/LLM (0.0 to 10.0).
        baseline_score: The pure mathematical score from Layer 1 MetricEngine.
        band: The maximum allowable deviation (default ±2.5 points).
        asymmetric: If True, upward adjustments are capped at (baseline + band),
            while downward adjustments are accepted in full (protecting against
            optimistic hallucinations).

    Returns:
        The anchored float score clamped between 0.0 and 10.0.
    """
    proposed_clamped = max(0.0, min(10.0, proposed_score))
    baseline_clamped = max(0.0, min(10.0, baseline_score))

    if asymmetric:
        # Upward adjustment is bounded to baseline + band; downward adjustment honored
        upper_limit = min(10.0, baseline_clamped + band)
        anchored = min(upper_limit, proposed_clamped)
    else:
        # Symmetric band: proposed score must lie in [baseline - band, baseline + band]
        lower_limit = max(0.0, baseline_clamped - band)
        upper_limit = min(10.0, baseline_clamped + band)
        anchored = max(lower_limit, min(upper_limit, proposed_clamped))

    return round(anchored, 2)


def graded_cap(severity: float, cap_floor: float, passing_threshold: float = 8.0) -> float:
    """Compute a score cap that scales proportionally with the severity of a guard breach.

    A fixed cap acts as an artificial cliff (e.g., extraction of 0.69 and 0.05 both
    collapsing to 3.0). This graded ramp ensures that marginal breaches retain distinction
    from catastrophic failures while strictly staying below passing_threshold.

    Parameters:
        severity: Normalized breach severity in [0.0, 1.0], where 1.0 is total failure.
        cap_floor: The lowest score for a total failure (e.g. 2.0 or 3.0).
        passing_threshold: The passing bar (default 8.0).

    Returns:
        Proportional score cap rounded to 2 decimal places.
    """
    sev = min(1.0, max(0.0, severity))
    cap_ceiling = max(cap_floor, min(cap_floor + 2.5, passing_threshold - 2.0))
    return round(cap_ceiling - (cap_ceiling - cap_floor) * sev, 2)


def escalate_disposition(current: Disposition | str | None, candidate: Disposition | str) -> Disposition:
    """Return the more severe of two dispositions.

    Order of severity: READY < CONDITIONAL < SUSPENDED.
    A downstream guard may escalate severity, but can never upgrade it.
    """
    cand_enum = candidate if isinstance(candidate, Disposition) else Disposition(candidate)
    if current is None:
        return cand_enum
    curr_enum = current if isinstance(current, Disposition) else Disposition(current)
    return cand_enum if cand_enum.severity > curr_enum.severity else curr_enum


def apply_fatal_flaw_overrides(
    score: float,
    disposition: Disposition,
    layer1_metrics: Layer1MetricResult,
    domain_votes: dict[str, DomainAgentVote],
    table_role: TableRole = "primary",
    target_column: str | None = None,
    applied_fixes_count: int = 0,
    config: AnchoringConfig | None = None,
) -> tuple[float, Disposition, list[str]]:
    """Evaluate deterministic fatal flaw override rules against the dataset.

    Returns:
        tuple of (overridden_score, overridden_disposition, list_of_applied_override_messages)
    """
    cfg = config or AnchoringConfig()
    current_score = score
    current_disp = disposition
    overrides: list[str] = []

    # 1. Fatal Flaw: Critical Unmasked PII (SSN, Credit Cards, Passports)
    # Critical PII applies to ALL tables regardless of role or use-case.
    detected_pii = layer1_metrics.detected_pii
    has_critical_pii = False
    critical_pii_cols: list[str] = []

    for item in detected_pii:
        col = str(item.get("column", "")).lower().strip()
        pii_type = str(item.get("type", "")).lower().strip()
        if pii_type in cfg.critical_pii_hints or col in cfg.critical_pii_hints:
            has_critical_pii = True
            critical_pii_cols.append(col or pii_type)

    if has_critical_pii:
        current_score = min(current_score, cfg.critical_pii_score_cap)
        current_disp = current_disp.escalate(Disposition.SUSPENDED)
        cols_str = ", ".join(sorted(set(critical_pii_cols)))
        overrides.append(
            f"[Fatal Flaw] Critical unmasked PII detected in column(s): {cols_str}. "
            f"Score capped at {cfg.critical_pii_score_cap} and disposition escalated to SUSPENDED."
        )

    # 2. Fatal Flaw: Target column completeness on primary tables
    is_primary = table_role == "primary"
    if is_primary and target_column:
        target_null_rate = layer1_metrics.null_rates.get(target_column, 0.0)
        if target_null_rate > cfg.target_null_max_rate:
            current_score = min(current_score, cfg.target_null_score_cap)
            current_disp = current_disp.escalate(Disposition.SUSPENDED)
            overrides.append(
                f"[Fatal Flaw] Primary target column '{target_column}' null rate is "
                f"{target_null_rate:.2%} (exceeds allowable {cfg.target_null_max_rate:.2%}). "
                f"Score capped at {cfg.target_null_score_cap}."
            )

    # 3. Schema Complexity Ceiling
    # Trivially narrow/homogenous tables (< 0.5 complexity) cannot score >= 8.0 (READY).
    if layer1_metrics.schema_complexity < cfg.schema_complexity_threshold:
        current_score = min(current_score, cfg.schema_complexity_score_cap)
        current_disp = current_disp.escalate(Disposition.CONDITIONAL)
        overrides.append(
            f"[Schema Guard] Schema complexity is {layer1_metrics.schema_complexity:.2f} "
            f"(< {cfg.schema_complexity_threshold:.2f}). Score capped at "
            f"{cfg.schema_complexity_score_cap} (CONDITIONAL ceiling)."
        )

    # 4. Business Alignment & Row Sufficiency (Primary tables only)
    biz_vote = domain_votes.get("business_mapping") or domain_votes.get("business_alignment")
    if is_primary and biz_vote:
        biz_conf = biz_vote.confidence
        data_fitness = str(biz_vote.metadata.get("data_fitness", "strong")).lower()
        if biz_conf < cfg.min_business_confidence or data_fitness == "poor":
            current_score = min(current_score, cfg.low_confidence_score_cap)
            current_disp = current_disp.escalate(Disposition.SUSPENDED)
            overrides.append(
                f"[Fatal Flaw] Low business mapping confidence ({biz_conf:.2f}) "
                f"or poor data fitness ('{data_fitness}'). Score capped at {cfg.low_confidence_score_cap}."
            )

        if layer1_metrics.total_rows < cfg.min_rows_ml:
            current_score = min(current_score, cfg.low_rows_score_cap)
            current_disp = current_disp.escalate(Disposition.CONDITIONAL)
            overrides.append(
                f"[Data Fitness] Row count ({layer1_metrics.total_rows}) is below ML threshold "
                f"({cfg.min_rows_ml}). Score capped at {cfg.low_rows_score_cap}."
            )

    # 5. No-Op Relaxation Prevention
    # If zero fixes have been executed, the consensus score cannot exceed baseline + 0.5.
    if applied_fixes_count == 0:
        max_allowed = layer1_metrics.overall_score + cfg.noop_relaxation_max_lift
        if current_score > max_allowed:
            capped_score = round(max_allowed, 2)
            overrides.append(
                f"[Anti-Relaxation] Zero fixes applied. Score restricted from {current_score:.2f} "
                f"to baseline + {cfg.noop_relaxation_max_lift} ({capped_score:.2f})."
            )
            current_score = capped_score

    # Escalate disposition based on domain votes
    for vote in domain_votes.values():
        if vote.disposition.severity > current_disp.severity:
            current_disp = current_disp.escalate(vote.disposition)

    return current_score, current_disp, overrides


def synthesize_consensus(
    layer1_metrics: Layer1MetricResult,
    domain_votes: dict[str, DomainAgentVote],
    table_role: TableRole = "primary",
    target_column: str | None = None,
    proposed_judge_score: float | None = None,
    applied_fixes_count: int = 0,
    config: AnchoringConfig | None = None,
) -> ConsensusVerdict:
    """Synthesize Layer 1 metrics and Layer 2 domain agent votes into a Layer 3 Consensus Verdict.

    Algorithm:
    1. Compute initial proposed score from Judge or weighted domain agent average.
    2. Anchor proposed score within ±2.5 of Layer 1 deterministic baseline.
    3. Aggregate category scores (quality, governance, schema, fairness).
    4. Apply deterministic fatal flaw overrides and graded score caps.
    5. Deduplicate and collect suggested actionable remediations.
    6. Derive final operational verdict (READY, CONDITIONALLY_READY, FAILED).
    """
    cfg = config or AnchoringConfig()
    baseline = layer1_metrics.overall_score

    # Determine candidate score
    if proposed_judge_score is not None:
        raw_candidate: float = float(proposed_judge_score)
    elif domain_votes:
        # Weighted mean of domain agent scores using confidence weights
        scored_votes = [v for v in domain_votes.values() if v.score is not None]
        if scored_votes:
            total_weight = sum(max(0.1, v.confidence) for v in scored_votes)
            raw_candidate = float(sum(float(v.score or 0.0) * max(0.1, v.confidence) for v in scored_votes) / total_weight)
        else:
            raw_candidate = float(baseline)
    else:
        raw_candidate = float(baseline)

    # Step 1: Anchor the score mathematically to the deterministic baseline
    anchored = anchor_score(raw_candidate, baseline, band=cfg.anchor_band, asymmetric=False)

    # Step 2: Determine initial disposition from agent votes
    initial_disp = Disposition.READY
    for vote in domain_votes.values():
        initial_disp = initial_disp.escalate(vote.disposition)

    # Step 3: Apply deterministic fatal flaw overrides and caps
    final_score, final_disp, overrides = apply_fatal_flaw_overrides(
        score=anchored,
        disposition=initial_disp,
        layer1_metrics=layer1_metrics,
        domain_votes=domain_votes,
        table_role=table_role,
        target_column=target_column,
        applied_fixes_count=applied_fixes_count,
        config=cfg,
    )

    # Step 4: Extract and synthesize category scores
    category_scores: dict[str, float] = {}
    for cat_name, cat_res in layer1_metrics.categories.items():
        category_scores[cat_name] = round(cat_res.score, 2)

    # Blend agent domain scores into categories where provided
    for vote in domain_votes.values():
        if vote.score is not None:
            domain_key = vote.domain.lower()
            if domain_key in category_scores:
                # Blend 60% deterministic Layer 1, 40% expert agent
                blended = (category_scores[domain_key] * 0.6) + (vote.score * 0.4)
                category_scores[domain_key] = round(blended, 2)
            else:
                category_scores[domain_key] = round(vote.score, 2)

    # Step 5: Collect and deduplicate suggestions
    seen_checks: set[str] = set()
    suggestions: list[ActionableFix] = []
    for vote in domain_votes.values():
        for fix in vote.suggested_fixes:
            if fix.addresses_check not in seen_checks:
                seen_checks.add(fix.addresses_check)
                suggestions.append(fix)

    # Step 6: Derive final verdict and passed status
    passed = (
        final_score >= cfg.passing_threshold
        and final_disp == Disposition.READY
        and len(overrides) == 0
    )

    if final_disp == Disposition.SUSPENDED or final_score < cfg.conditional_threshold:
        verdict = VerdictStatus.FAILED
    elif final_score >= cfg.passing_threshold and final_disp == Disposition.READY:
        verdict = VerdictStatus.READY
    else:
        verdict = VerdictStatus.CONDITIONALLY_READY

    explanation_parts: list[str] = [
        f"Consensus readiness evaluated at {final_score:.2f}/10.0 (baseline: {baseline:.2f}).",
        f"Final disposition: {final_disp.value.upper()}.",
    ]
    if overrides:
        explanation_parts.append(f"{len(overrides)} deterministic override(s) enforced.")
    if suggestions:
        explanation_parts.append(f"{len(suggestions)} remediation suggestion(s) proposed.")

    return ConsensusVerdict(
        verdict=verdict,
        global_readiness_score=round(final_score, 2),
        deterministic_baseline_score=round(baseline, 2),
        disposition=final_disp,
        category_scores=category_scores,
        consensus_explanation=" ".join(explanation_parts),
        overrides_applied=overrides,
        suggestions=suggestions,
        agent_votes=domain_votes,
        passed=passed,
    )
