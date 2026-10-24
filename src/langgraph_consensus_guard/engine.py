"""Layer 1 deterministic mathematical metric evaluation engine.

Computes pure arithmetic calculations, Shannon entropy, PII pattern detection,
and schema complexity on datasets without any LLM dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from langgraph_consensus_guard.schema import (
    Layer1MetricResult,
    MetricCategoryResult,
    MetricCheck,
)

# Standard regexes for PII detection
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "SSN": re.compile(r"^(?:\d{3}-\d{2}-\d{4}|\d{9})$"),
    "CREDIT_CARD": re.compile(r"^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})$"),
    "EMAIL": re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
    "PHONE": re.compile(r"^\+?[1-9]\d{1,14}$|^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$"),
    "PASSPORT": re.compile(r"^[A-Z][0-9]{8}$"),
}

CRITICAL_PII_TYPES = frozenset({"SSN", "CREDIT_CARD", "PASSPORT"})


def compute_shannon_entropy(values: list[Any]) -> float:
    """Compute Shannon entropy in bits for a sequence of discrete values."""
    if not values:
        return 0.0
    total = len(values)
    counts = Counter(str(v) for v in values if v is not None)
    if not counts:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def evaluate_records(
    records: list[dict[str, Any]],
    dataset_ref: str = "dataset",
    run_id: str = "run_001",
    passing_threshold: float = 8.0,
) -> Layer1MetricResult:
    """Evaluate a tabular dataset (list of row dictionaries) mathematically.

    Computes Layer 1 metrics:
    - Quality: null rates, duplicate rows, row completeness.
    - Governance: PII detection, Shannon entropy for synthetic/constant leakage.
    - Schema: column count, type conformance, schema complexity.
    """
    total_rows = len(records)
    if total_rows == 0:
        return Layer1MetricResult(
            run_id=run_id,
            dataset_ref=dataset_ref,
            overall_score=0.0,
            passed=False,
            categories={},
            total_rows=0,
            metadata={"empty_dataset": True},
        )

    # 1. Identify all column names
    columns: list[str] = sorted({col for r in records for col in r})
    col_count = len(columns)

    # 2. Quality Metrics: Null rates & duplicates
    null_counts: dict[str, int] = dict.fromkeys(columns, 0)
    col_values: dict[str, list[Any]] = {col: [] for col in columns}

    for row in records:
        for col in columns:
            val = row.get(col)
            col_values[col].append(val)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                null_counts[col] += 1

    null_rates: dict[str, float] = {
        col: round(null_counts[col] / total_rows, 4) for col in columns
    }
    avg_null_rate = sum(null_rates.values()) / max(1, col_count)
    completeness_score = max(0.0, 10.0 * (1.0 - avg_null_rate))

    # Duplicate row calculation
    row_tuples = [tuple(sorted((k, str(v)) for k, v in r.items())) for r in records]
    unique_rows = len(set(row_tuples))
    duplicate_rows = total_rows - unique_rows
    duplicate_rate = duplicate_rows / total_rows
    duplicate_score = max(0.0, 10.0 * (1.0 - (duplicate_rate * 2.0)))

    quality_score = round((completeness_score * 0.7) + (duplicate_score * 0.3), 2)
    quality_checks = [
        MetricCheck(
            name="completeness_score",
            passed=avg_null_rate <= 0.10,
            score=round(completeness_score, 2),
            value=round(1.0 - avg_null_rate, 4),
            threshold=0.90,
            detail=f"Average column completeness is {(1.0 - avg_null_rate):.2%}.",
        ),
        MetricCheck(
            name="duplicate_rate",
            passed=duplicate_rate <= 0.05,
            score=round(duplicate_score, 2),
            value=round(duplicate_rate, 4),
            threshold=0.05,
            detail=f"{duplicate_rows} duplicate rows detected ({duplicate_rate:.2%}).",
        ),
    ]

    # 3. Governance Metrics: PII & Entropy
    detected_pii: list[dict[str, Any]] = []
    low_entropy_cols: list[str] = []

    for col in columns:
        vals = col_values[col]
        # PII regex scan on non-null sample
        sample_vals = [str(v).strip() for v in vals if v is not None][:100]
        for pii_type, pattern in PII_PATTERNS.items():
            matches = sum(1 for v in sample_vals if pattern.match(v))
            if sample_vals and (matches / len(sample_vals)) >= 0.2:
                detected_pii.append(
                    {
                        "column": col,
                        "type": pii_type,
                        "match_rate": round(matches / len(sample_vals), 2),
                        "sample_count": len(sample_vals),
                    }
                )

        # Shannon entropy check (on columns with > 5 unique non-null values)
        non_null_vals = [v for v in vals if v is not None]
        if len({str(v) for v in non_null_vals}) > 5:
            ent = compute_shannon_entropy(non_null_vals)
            if ent < 1.0:
                low_entropy_cols.append(col)

    has_critical = any(
        str(item.get("type", "")).upper() in CRITICAL_PII_TYPES for item in detected_pii
    )
    if has_critical:
        gov_score = 2.0
        gov_passed = False
    elif detected_pii:
        gov_score = 6.0
        gov_passed = True
    else:
        gov_score = 10.0
        gov_passed = True

    gov_checks = [
        MetricCheck(
            name="pii_detection",
            passed=not has_critical,
            score=gov_score,
            value=len(detected_pii),
            threshold=0,
            detail=f"{len(detected_pii)} PII pattern match(es) identified.",
        )
    ]

    # 4. Schema Complexity Metrics
    # Narrow schemas (< 3 columns) receive a structural penalty
    if col_count < 2:
        schema_complexity = 0.2
        schema_score = 3.0
    elif col_count < 4:
        schema_complexity = 0.45
        schema_score = 6.5
    else:
        schema_complexity = min(1.0, 0.5 + (col_count * 0.05))
        schema_score = min(10.0, 5.0 + (schema_complexity * 5.0))

    schema_score = round(schema_score, 2)
    schema_checks = [
        MetricCheck(
            name="schema_complexity",
            passed=schema_complexity >= 0.5,
            score=schema_score,
            value=round(schema_complexity, 2),
            threshold=0.5,
            detail=f"Dataset contains {col_count} columns with complexity {schema_complexity:.2f}.",
        )
    ]

    # 5. Aggregate Categories & Overall Score
    categories: dict[str, MetricCategoryResult] = {
        "quality": MetricCategoryResult(
            category="quality",
            score=quality_score,
            passed=quality_score >= passing_threshold,
            checks=quality_checks,
            failures=[c.name for c in quality_checks if not c.passed],
        ),
        "governance": MetricCategoryResult(
            category="governance",
            score=gov_score,
            passed=gov_passed,
            checks=gov_checks,
            failures=[c.name for c in gov_checks if not c.passed],
        ),
        "schema": MetricCategoryResult(
            category="schema",
            score=schema_score,
            passed=schema_score >= passing_threshold,
            checks=schema_checks,
            failures=[c.name for c in schema_checks if not c.passed],
        ),
    }

    # Deterministic weighted calculation: 45% Quality, 40% Governance, 15% Schema
    overall_score = round(
        (quality_score * 0.45) + (gov_score * 0.40) + (schema_score * 0.15), 2
    )
    overall_passed = overall_score >= passing_threshold and gov_passed

    return Layer1MetricResult(
        run_id=run_id,
        dataset_ref=dataset_ref,
        overall_score=overall_score,
        passed=overall_passed,
        categories=categories,
        detected_pii=detected_pii,
        null_rates=null_rates,
        duplicate_rate=round(duplicate_rate, 4),
        schema_complexity=round(schema_complexity, 2),
        total_rows=total_rows,
        metadata={"column_count": col_count, "low_entropy_columns": low_entropy_cols},
    )
