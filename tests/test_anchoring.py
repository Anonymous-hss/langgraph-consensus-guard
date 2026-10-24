"""Tests for deterministic anchoring, fatal flaw overrides, and consensus synthesis."""

from __future__ import annotations

from langgraph_consensus_guard.anchoring import (
    anchor_score,
    apply_fatal_flaw_overrides,
    escalate_disposition,
    graded_cap,
    synthesize_consensus,
)
from langgraph_consensus_guard.schema import (
    ActionableFix,
    Disposition,
    DomainAgentVote,
    Layer1MetricResult,
    VerdictStatus,
)


def test_anchor_score_symmetric() -> None:
    """Verify that an LLM proposed score is strictly anchored within ±2.5 of baseline."""
    baseline = 6.0

    # Proposed 9.5 on a 6.0 baseline must be clamped to 8.5
    assert anchor_score(9.5, baseline, band=2.5) == 8.5

    # Proposed 2.0 on a 6.0 baseline must be clamped to 3.5
    assert anchor_score(2.0, baseline, band=2.5) == 3.5

    # Proposed 7.0 is within [3.5, 8.5], so remains 7.0
    assert anchor_score(7.0, baseline, band=2.5) == 7.0

    # Edge cases
    assert anchor_score(10.0, 10.0, band=2.5) == 10.0
    assert anchor_score(0.0, 0.0, band=2.5) == 0.0


def test_anchor_score_asymmetric() -> None:
    """Verify asymmetric anchoring (upward capped, downward fully honored)."""
    baseline = 7.0

    # Upward adjustment capped at 7.0 + 2.0 = 9.0
    assert anchor_score(9.8, baseline, band=2.0, asymmetric=True) == 9.0

    # Downward adjustment accepted in full (e.g., agent identifies severe flaw)
    assert anchor_score(3.2, baseline, band=2.0, asymmetric=True) == 3.2


def test_graded_cap() -> None:
    """Verify that graded_cap ramps smoothly between ceiling and floor based on severity."""
    # Severity 0.0 -> cap_ceiling (floor + 2.5 bounded)
    cap_zero = graded_cap(0.0, cap_floor=3.0, passing_threshold=8.0)
    assert cap_zero == 5.5

    # Severity 1.0 (total failure) -> cap_floor (3.0)
    cap_full = graded_cap(1.0, cap_floor=3.0, passing_threshold=8.0)
    assert cap_full == 3.0

    # Severity 0.5 -> intermediate value
    cap_mid = graded_cap(0.5, cap_floor=3.0, passing_threshold=8.0)
    assert 3.0 < cap_mid < 5.5


def test_escalate_disposition() -> None:
    """Verify severity escalation between dispositions."""
    assert escalate_disposition(None, Disposition.READY) == Disposition.READY
    assert escalate_disposition(Disposition.READY, Disposition.CONDITIONAL) == Disposition.CONDITIONAL
    assert escalate_disposition(Disposition.CONDITIONAL, Disposition.READY) == Disposition.CONDITIONAL
    assert escalate_disposition(Disposition.CONDITIONAL, Disposition.SUSPENDED) == Disposition.SUSPENDED


def test_fatal_flaw_critical_pii() -> None:
    """Verify that critical PII caps score at 2.0 and forces SUSPENDED disposition."""
    l1 = Layer1MetricResult(
        run_id="run_pii",
        dataset_ref="users_table",
        overall_score=8.5,
        passed=True,
        detected_pii=[{"column": "ssn", "type": "SSN"}],
    )
    domain_votes = {
        "gov": DomainAgentVote(
            agent_id="gov",
            domain="governance",
            disposition=Disposition.READY,
            score=9.0,
            summary="All good.",
        )
    }

    score, disp, overrides = apply_fatal_flaw_overrides(
        score=8.5,
        disposition=Disposition.READY,
        layer1_metrics=l1,
        domain_votes=domain_votes,
        table_role="primary",
    )

    assert score == 2.0
    assert disp == Disposition.SUSPENDED
    assert any("Critical unmasked PII" in msg for msg in overrides)


def test_fatal_flaw_target_completeness_table_roles() -> None:
    """Verify target null rate applies ONLY to primary tables, not reference tables."""
    l1 = Layer1MetricResult(
        run_id="run_target",
        dataset_ref="orders",
        overall_score=8.0,
        passed=True,
        null_rates={"label": 0.12},  # 12% nulls (> 5%)
    )

    # 1. Primary table with target column -> Capped at 3.0, SUSPENDED
    score_p, disp_p, overrides_p = apply_fatal_flaw_overrides(
        score=8.0,
        disposition=Disposition.READY,
        layer1_metrics=l1,
        domain_votes={},
        table_role="primary",
        target_column="label",
    )
    assert score_p == 3.0
    assert disp_p == Disposition.SUSPENDED
    assert any("Primary target column" in msg for msg in overrides_p)

    # 2. Reference/lookup table with target column -> NOT penalized
    score_r, disp_r, overrides_r = apply_fatal_flaw_overrides(
        score=8.0,
        disposition=Disposition.READY,
        layer1_metrics=l1,
        domain_votes={},
        table_role="reference",
        target_column="label",
    )
    assert score_r == 8.0
    assert disp_r == Disposition.READY
    assert len(overrides_r) == 0


def test_schema_complexity_ceiling() -> None:
    """Verify that narrow/trivial schemas (< 0.5 complexity) are capped below 8.0."""
    l1 = Layer1MetricResult(
        run_id="run_narrow",
        dataset_ref="narrow_csv",
        overall_score=9.0,
        passed=True,
        schema_complexity=0.35,  # Narrow schema
    )

    score, disp, overrides = apply_fatal_flaw_overrides(
        score=9.0,
        disposition=Disposition.READY,
        layer1_metrics=l1,
        domain_votes={},
        table_role="primary",
    )
    assert score == 7.9
    assert disp == Disposition.CONDITIONAL
    assert any("Schema complexity" in msg for msg in overrides)


def test_noop_relaxation_prevention() -> None:
    """Verify that if zero fixes have been executed, score cannot exceed baseline + 0.5."""
    l1 = Layer1MetricResult(
        run_id="run_relax",
        dataset_ref="data",
        overall_score=6.0,
        passed=False,
    )

    # Agent/Judge attempts to propose 8.0 with zero fixes applied
    score, _disp, overrides = apply_fatal_flaw_overrides(
        score=8.0,
        disposition=Disposition.READY,
        layer1_metrics=l1,
        domain_votes={},
        applied_fixes_count=0,
    )
    assert score == 6.5
    assert any("Anti-Relaxation" in msg for msg in overrides)


def test_synthesize_consensus_full_flow() -> None:
    """Verify end-to-end consensus verdict synthesis."""
    l1 = Layer1MetricResult(
        run_id="run_e2e",
        dataset_ref="public.customers",
        overall_score=8.2,
        passed=True,
        schema_complexity=0.8,
        total_rows=500,
    )
    fix = ActionableFix(
        type="quality",
        addresses_check="null_rate:phone",
        description="Impute missing phone numbers with unknown tag.",
        auto_apply=True,
        risk_level="Low",
    )
    votes = {
        "quality_analyst": DomainAgentVote(
            agent_id="quality_analyst",
            domain="quality",
            disposition=Disposition.READY,
            score=8.6,
            summary="Quality criteria satisfied.",
            suggested_fixes=[fix],
        ),
        "governance_guardian": DomainAgentVote(
            agent_id="governance_guardian",
            domain="governance",
            disposition=Disposition.READY,
            score=8.8,
            summary="Zero PII violations.",
        ),
    }

    verdict = synthesize_consensus(
        layer1_metrics=l1,
        domain_votes=votes,
        table_role="primary",
        applied_fixes_count=1,
    )

    assert verdict.verdict == VerdictStatus.READY
    assert verdict.disposition == Disposition.READY
    assert verdict.passed is True
    assert 8.0 <= verdict.global_readiness_score <= 9.0
    assert len(verdict.suggestions) == 1
    assert verdict.suggestions[0].addresses_check == "null_rate:phone"
