"""Tests for core state schema models, serialization, and reducers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from langgraph_consensus_guard.schema import (
    ActionableFix,
    ConsensusVerdict,
    Disposition,
    DomainAgentVote,
    Layer1MetricResult,
    MetricCategoryResult,
    MetricCheck,
    VerdictStatus,
    merge_votes,
)


def test_disposition_severity_and_escalate() -> None:
    """Verify Disposition enum severity hierarchy and escalation logic."""
    assert Disposition.READY.severity == 0
    assert Disposition.CONDITIONAL.severity == 1
    assert Disposition.SUSPENDED.severity == 2

    # Escalation cannot downgrade
    assert Disposition.READY.escalate(Disposition.CONDITIONAL) == Disposition.CONDITIONAL
    assert Disposition.CONDITIONAL.escalate(Disposition.SUSPENDED) == Disposition.SUSPENDED
    assert Disposition.SUSPENDED.escalate(Disposition.READY) == Disposition.SUSPENDED
    assert Disposition.SUSPENDED.escalate(Disposition.CONDITIONAL) == Disposition.SUSPENDED
    assert Disposition.READY.escalate("suspended") == Disposition.SUSPENDED


def test_actionable_fix_model() -> None:
    """Verify ActionableFix instantiation and validation."""
    fix = ActionableFix(
        type="quality",
        addresses_check="null_rate:price",
        description="Impute null values in price column using median.",
        usecase_validation="Preserves retail revenue calculations.",
        impact_assessment="Zero rows dropped; fills 12 missing values.",
        auto_apply=True,
        risk_level="Low",
        code="df['price'] = df['price'].fillna(df['price'].median())",
    )
    assert fix.type == "quality"
    assert fix.auto_apply is True
    assert fix.risk_level == "Low"

    data = fix.model_dump()
    assert data["addresses_check"] == "null_rate:price"


def test_layer1_metric_result_validation() -> None:
    """Verify Layer1MetricResult construction, category lookup, and PII checks."""
    check = MetricCheck(
        name="completeness_score",
        passed=True,
        score=9.5,
        value=0.98,
        threshold=0.90,
        detail="High completeness.",
    )
    category = MetricCategoryResult(
        category="quality",
        score=9.2,
        passed=True,
        checks=[check],
        failures=[],
    )
    l1 = Layer1MetricResult(
        run_id="run_123",
        dataset_ref="public.orders",
        overall_score=8.7,
        passed=True,
        categories={"quality": category},
        detected_pii=[{"column": "customer_ssn", "type": "SSN"}],
        null_rates={"price": 0.02, "order_id": 0.0},
        duplicate_rate=0.01,
        schema_complexity=0.85,
        total_rows=1500,
    )

    assert l1.get_category_score("quality") == 9.2
    assert l1.get_category_score("nonexistent", default=4.0) == 4.0
    assert l1.has_critical_pii() is True

    # Test score bounds validation
    with pytest.raises(ValidationError):
        Layer1MetricResult(
            run_id="bad_run",
            dataset_ref="test",
            overall_score=11.5,  # Invalid: > 10.0
            passed=False,
        )


def test_domain_agent_vote_model() -> None:
    """Verify DomainAgentVote structure and serialization."""
    vote = DomainAgentVote(
        agent_id="governance_guardian",
        domain="governance",
        disposition=Disposition.CONDITIONAL,
        score=7.0,
        confidence=0.95,
        summary="Minor unmasked email addresses detected in metadata.",
        findings=["Column 'contact_email' contains raw emails."],
        suggested_fixes=[
            ActionableFix(
                type="governance",
                addresses_check="pii_detection:contact_email",
                description="Hash or mask email addresses.",
                auto_apply=False,
                risk_level="Medium",
            )
        ],
    )
    assert vote.disposition == Disposition.CONDITIONAL
    assert len(vote.suggested_fixes) == 1
    dump = vote.model_dump()
    assert dump["agent_id"] == "governance_guardian"


def test_consensus_verdict_model() -> None:
    """Verify ConsensusVerdict fields and passed state."""
    verdict = ConsensusVerdict(
        verdict=VerdictStatus.READY,
        global_readiness_score=8.6,
        deterministic_baseline_score=8.2,
        disposition=Disposition.READY,
        category_scores={"quality": 8.8, "governance": 8.5},
        consensus_explanation="High quality dataset cleared for production ML pipeline.",
        overrides_applied=[],
        suggestions=[],
        passed=True,
    )
    assert verdict.passed is True
    assert verdict.verdict == VerdictStatus.READY
    assert verdict.disposition == Disposition.READY


def test_merge_votes_reducer() -> None:
    """Verify the LangGraph state reducer for merging agent votes."""
    vote1 = DomainAgentVote(
        agent_id="quality_agent",
        domain="quality",
        disposition=Disposition.READY,
        score=9.0,
        summary="Quality passed.",
    )
    vote2 = DomainAgentVote(
        agent_id="governance_agent",
        domain="governance",
        disposition=Disposition.CONDITIONAL,
        score=7.5,
        summary="Governance warning.",
    )

    merged = merge_votes(None, [vote1])
    assert len(merged) == 1
    assert "quality_agent" in merged

    merged_dict = merge_votes(merged, {"governance_agent": vote2})
    assert len(merged_dict) == 2
    assert merged_dict["governance_agent"].disposition == Disposition.CONDITIONAL
