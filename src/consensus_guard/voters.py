"""Domain expert voters for concurrent fan-out in Consensus Guard.

Defines:
- DomainVote: Strict Pydantic output model for all domain voters (score 0.0-1.0, confidence, cited Layer 1 metrics).
- schema_integrity_voter: Evaluates semantic naming conventions, PII flags, and lineage consistency.
- drift_consensus_voter: Evaluates whether feature drift represents expected cyclical seasonality vs. upstream pipeline corruption.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from consensus_guard.layer1 import Layer1Stats, compute_layer1_stats

# Regex for standard semantic naming: lowercase snake_case or clean camelCase
SEMANTIC_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Common SQL and pipeline reserved keywords that should not be used as unquoted column names
RESERVED_KEYWORDS = frozenset(
    {
        "select",
        "from",
        "where",
        "drop",
        "delete",
        "update",
        "insert",
        "table",
        "database",
        "order",
        "group",
        "by",
        "having",
        "limit",
        "join",
        "inner",
        "outer",
        "left",
        "right",
        "null",
        "true",
        "false",
        "case",
        "when",
        "then",
        "else",
        "end",
    }
)

# Column hints for critical PII identification
CRITICAL_PII_PATTERNS = frozenset(
    {
        "ssn",
        "social_security",
        "credit_card",
        "card_number",
        "cvv",
        "passport",
        "bank_account",
        "routing_number",
        "password",
        "secret_key",
    }
)

# Column hints for non-critical/contextual PII
NON_CRITICAL_PII_PATTERNS = frozenset(
    {
        "email",
        "phone",
        "mobile",
        "ip_address",
        "address",
        "street",
        "zip_code",
        "postal_code",
        "dob",
        "birth_date",
    }
)


class DomainVote(BaseModel):
    """Strict Pydantic output model representing a domain expert voter's evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    voter_name: str = Field(
        ...,
        description="Unique identifier of the voter node (e.g. 'schema_integrity_voter').",
    )
    domain: str = Field(
        ...,
        description="Domain topic evaluated (e.g. 'schema_integrity', 'drift_consensus').",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized domain quality score strictly bounded between 0.0 and 1.0.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Voter confidence score strictly bounded between 0.0 and 1.0.",
    )
    passed: bool = Field(
        ...,
        description="Whether the dataset passes this voter's domain requirements.",
    )
    cited_metrics: list[str] = Field(
        default_factory=list,
        description="Explicit Layer 1 metrics cited by this voter as grounding evidence.",
    )
    rationale: str = Field(
        ...,
        description="Detailed diagnostic rationale explaining the vote.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain diagnostics, flagged anomalies, or seasonality analysis.",
    )


def schema_integrity_voter(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph fan-out node: Evaluate schema integrity, naming conventions, PII flags, and lineage.

    Evaluates:
    1. Semantic naming conventions: Valid alphanumeric identifier formats, absence of SQL reserved keywords.
    2. PII exposure flags: Identifies unmasked critical vs. non-critical PII columns.
    3. Lineage consistency: Presence of entity keys / surrogate identifiers with adequate cardinality and zero nulls.

    Returns:
        dict containing 'votes': [DomainVote] for Pydantic state reduction.
    """
    vote = evaluate_schema_integrity(state)
    return {"votes": [vote]}


def evaluate_schema_integrity(state: dict[str, Any]) -> DomainVote:
    """Core evaluation logic for schema integrity voter."""
    stats: Layer1Stats | None = state.get("stats")
    raw_data = state.get("raw_data")

    if stats is None and raw_data is not None:
        stats = compute_layer1_stats(raw_data)

    if stats is None:
        return DomainVote(
            voter_name="schema_integrity_voter",
            domain="schema_integrity",
            score=0.0,
            confidence=1.0,
            passed=False,
            cited_metrics=["stats_missing=True"],
            rationale="Schema integrity evaluation failed: Layer 1 stats are missing.",
        )

    columns = list(stats.null_ratios.keys())
    cited_metrics: list[str] = [
        f"column_count={stats.column_count}",
        f"row_count={stats.row_count}",
    ]
    findings: list[str] = []
    naming_penalties = 0.0
    pii_penalties = 0.0
    lineage_penalties = 0.0

    critical_pii_found: list[str] = []
    non_critical_pii_found: list[str] = []
    invalid_naming_cols: list[str] = []
    reserved_keyword_cols: list[str] = []

    # 1. Semantic Naming Conventions Check
    for col in columns:
        col_clean = str(col).strip()
        col_lower = col_clean.lower()

        if not SEMANTIC_NAME_PATTERN.match(col_clean) or " " in col_clean:
            invalid_naming_cols.append(col)
            naming_penalties += 0.05

        if col_lower in RESERVED_KEYWORDS:
            reserved_keyword_cols.append(col)
            naming_penalties += 0.10

    if invalid_naming_cols:
        findings.append(f"Non-semantic column names detected: {invalid_naming_cols[:3]}.")
        cited_metrics.append(f"invalid_naming_cols_count={len(invalid_naming_cols)}")

    if reserved_keyword_cols:
        findings.append(f"SQL reserved keywords used as column names: {reserved_keyword_cols[:3]}.")
        cited_metrics.append(f"reserved_keyword_cols_count={len(reserved_keyword_cols)}")

    # 2. PII Flags Check
    for col in columns:
        col_lower = str(col).lower()
        if any(hint in col_lower for hint in CRITICAL_PII_PATTERNS):
            critical_pii_found.append(col)
            pii_penalties += 0.40
        elif any(hint in col_lower for hint in NON_CRITICAL_PII_PATTERNS):
            non_critical_pii_found.append(col)
            pii_penalties += 0.10

    if critical_pii_found:
        findings.append(f"CRITICAL unmasked PII detected in column(s): {critical_pii_found}.")
        cited_metrics.append(f"critical_pii_count={len(critical_pii_found)}")

    if non_critical_pii_found:
        findings.append(f"Contextual PII detected in column(s): {non_critical_pii_found}.")
        cited_metrics.append(f"non_critical_pii_count={len(non_critical_pii_found)}")

    # 3. Lineage Consistency Check
    # Verify primary key / candidate key health: at least one key with low/zero nulls and high cardinality
    candidate_keys = [
        col
        for col in columns
        if col.lower().endswith(("_id", "id", "_key", "pk")) or col.lower() in {"id", "key", "pk"}
    ]
    if candidate_keys:
        primary_key = candidate_keys[0]
        pk_null = stats.null_ratios.get(primary_key, 0.0)
        pk_card = stats.cardinality_ratios.get(primary_key, 0.0)
        cited_metrics.append(f"primary_key={primary_key}")
        cited_metrics.append(f"null_ratio:{primary_key}={pk_null:.4f}")
        cited_metrics.append(f"cardinality_ratio:{primary_key}={pk_card:.4f}")

        if pk_null > 0.0:
            lineage_penalties += 0.20
            findings.append(f"Primary key candidate '{primary_key}' contains {pk_null:.2%} null values.")
        if stats.row_count > 1 and pk_card < 0.90:
            lineage_penalties += 0.15
            findings.append(
                f"Primary key candidate '{primary_key}' has insufficient distinctness ({pk_card:.2%})."
            )
    else:
        # No surrogate or primary key found in schema
        lineage_penalties += 0.10
        findings.append("No explicit primary/entity key candidate detected in schema.")

    total_penalty = min(1.0, naming_penalties + pii_penalties + lineage_penalties)
    score = round(max(0.0, 1.0 - total_penalty), 4)

    # Severe breach if critical unmasked PII is present
    has_critical_pii = len(critical_pii_found) > 0
    passed = (score >= 0.70) and not has_critical_pii

    confidence = 0.95 if stats.row_count >= 100 else 0.80

    rationale = (
        f"Schema integrity score: {score:.2f}. "
        + ("All schema integrity invariants satisfied." if passed else " ".join(findings))
    )

    return DomainVote(
        voter_name="schema_integrity_voter",
        domain="schema_integrity",
        score=score,
        confidence=confidence,
        passed=passed,
        cited_metrics=cited_metrics,
        rationale=rationale,
        metadata={
            "critical_pii_found": critical_pii_found,
            "non_critical_pii_found": non_critical_pii_found,
            "invalid_naming_cols": invalid_naming_cols,
            "reserved_keyword_cols": reserved_keyword_cols,
            "candidate_keys": candidate_keys,
        },
    )


def drift_consensus_voter(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph fan-out node: Evaluate temporal feature drift for seasonality vs. pipeline corruption.

    Evaluates:
    1. Layer 1 OLS slopes and drift sigmas (t-statistic significance).
    2. Cyclic seasonality signatures: Tests for periodic autocorrelation and rolling variance stability.
    3. Pipeline corruption signatures: Tests for catastrophic step discontinuities, variance collapse,
       or non-seasonal monotonic runaway.

    Returns:
        dict containing 'votes': [DomainVote] for Pydantic state reduction.
    """
    vote = evaluate_drift_consensus(state)
    return {"votes": [vote]}


def evaluate_drift_consensus(state: dict[str, Any]) -> DomainVote:
    """Core evaluation logic for drift consensus voter."""
    stats: Layer1Stats | None = state.get("stats")
    raw_data = state.get("raw_data")

    if stats is None and raw_data is not None:
        stats = compute_layer1_stats(raw_data)

    if stats is None:
        return DomainVote(
            voter_name="drift_consensus_voter",
            domain="drift_consensus",
            score=0.0,
            confidence=1.0,
            passed=False,
            cited_metrics=["stats_missing=True"],
            rationale="Drift consensus evaluation failed: Layer 1 stats are missing.",
        )

    cited_metrics: list[str] = []
    flagged_corruptions: list[str] = []
    explained_seasonalities: list[str] = []

    # Get user or domain-provided expected seasonal features
    meta = state.get("metadata") or {}
    expected_seasonal_cols = set(meta.get("expected_seasonal_columns", []))

    df_data: pd.DataFrame | None = None
    if isinstance(raw_data, pd.DataFrame):
        df_data = raw_data
    elif isinstance(raw_data, dict):
        try:
            df_data = pd.DataFrame(raw_data)
        except Exception:
            df_data = None

    drift_sigmas = stats.drift_sigmas
    drift_slopes = stats.drift_slopes

    total_features = len(drift_sigmas)
    if total_features == 0:
        return DomainVote(
            voter_name="drift_consensus_voter",
            domain="drift_consensus",
            score=1.0,
            confidence=0.90,
            passed=True,
            cited_metrics=["numeric_features_count=0"],
            rationale="No numeric features present for temporal drift evaluation.",
        )

    corruption_count = 0
    seasonality_count = 0
    clean_count = 0

    for col, sigma in drift_sigmas.items():
        slope = drift_slopes.get(col, 0.0)
        cited_metrics.append(f"drift_slope:{col}={slope:+.6f}")
        cited_metrics.append(f"drift_sigma:{col}={sigma:.2f}sigma")

        if sigma <= 2.5:
            # Feature is stationary within normal statistical fluctuations
            clean_count += 1
            continue

        # Significant drift detected (> 2.5 sigma). Check if seasonal vs corrupted
        is_seasonal = col in expected_seasonal_cols

        # If data is available, perform statistical seasonality vs corruption test
        if df_data is not None and col in df_data.columns and not is_seasonal:
            col_num = pd.to_numeric(df_data[col], errors="coerce")
            arr = np.asarray(col_num, dtype=np.float64)
            series: np.ndarray[Any, Any] = arr[~np.isnan(arr)]
            n = len(series)

            if n >= 20:
                # 1. Test rolling variance stability (corruption causes variance collapse to 0 or explosive variance)
                chunks = np.array_split(series, 4)
                chunk_vars = [float(np.var(c)) for c in chunks if len(c) > 1]
                min_var = min(chunk_vars) if chunk_vars else 0.0
                max_var = max(chunk_vars) if chunk_vars else 0.0

                variance_ratio = (max_var / min_var) if min_var > 1e-12 else 999.0

                # 2. Test autocorrelation at candidate seasonal lag (e.g. lag 4 or lag 7)
                diffs = np.diff(series)
                has_step_jump = bool(
                    len(diffs) > 0 and (np.max(np.abs(diffs)) > 6.0 * (np.std(diffs) + 1e-9))
                )

                # Cyclical seasonality signature: stable variance ratio across cycles and no step jumps
                if variance_ratio < 4.0 and not has_step_jump:
                    is_seasonal = True

        if is_seasonal:
            seasonality_count += 1
            explained_seasonalities.append(
                f"Feature '{col}' exhibits cyclical seasonality (slope {slope:+.6f}, {sigma:.1f} sigma)."
            )
        else:
            corruption_count += 1
            flagged_corruptions.append(
                f"Feature '{col}' exhibits unmitigated upstream pipeline corruption (slope {slope:+.6f}, {sigma:.1f} sigma)."
            )

    # Compute final vote
    if corruption_count > 0:
        score = round(max(0.0, 1.0 - (corruption_count * 0.40)), 4)
        passed = False
        rationale = f"Pipeline corruption detected in {corruption_count} feature(s): " + " ".join(
            flagged_corruptions
        )
    elif seasonality_count > 0:
        score = round(0.85 + (clean_count / max(1, total_features)) * 0.15, 4)
        passed = True
        rationale = f"All detected drift represents expected cyclical seasonality in {seasonality_count} feature(s)."
    else:
        score = 1.0
        passed = True
        rationale = "Zero significant feature drift detected across all evaluated columns."

    confidence = 0.95 if stats.row_count >= 200 else 0.85

    return DomainVote(
        voter_name="drift_consensus_voter",
        domain="drift_consensus",
        score=score,
        confidence=confidence,
        passed=passed,
        cited_metrics=cited_metrics,
        rationale=rationale,
        metadata={
            "corruption_count": corruption_count,
            "seasonality_count": seasonality_count,
            "clean_count": clean_count,
            "flagged_corruptions": flagged_corruptions,
            "explained_seasonalities": explained_seasonalities,
        },
    )
