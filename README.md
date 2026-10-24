# langgraph-consensus-guard

> Multi-agent consensus guardrails with deterministic mathematical anchoring for LangGraph workflows.

[![PyPI version](https://img.shields.io/badge/pypi-0.1.0-blue.svg)](https://pypi.org/project/langgraph-consensus-guard/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Type Checking: Strict Pyright](https://img.shields.io/badge/type_checker-pyright_strict-brightgreen.svg)](https://github.com/microsoft/pyright)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## The Problem: Hallucinated Optimism in Agentic Guardrails

Autonomous multi-agent workflows (e.g. data curation, code remediation, document parsing) frequently suffer from **hallucinated consensus**:
1. **Unanchored LLM Judges**: A qualitative LLM judge can be persuaded to score a failing dataset 9.5/10 despite widespread nulls or schema corruption.
2. **Artificial Cliff Caps**: Hard fixed caps collapse distinct failure modes (e.g., an extraction failure of 0.69 vs 0.05 both landing at 3.0), obliterating distribution nuance.
3. **Downstream Disposition Downgrades**: A lenient downstream agent erroneously upgrades a `SUSPENDED` corpus to `READY`.
4. **No-Op Relaxation**: Without anchoring to prior iterations, an agent can report a higher score after applying zero actual fixes.

`langgraph-consensus-guard` establishes an immutable, mathematically anchored guardrail pipeline directly within LangGraph. Qualitative domain agents can debate and add nuance, but **they cannot overrule mathematical facts**.

---

## 3-Layer Guardrail Architecture

```
                                [Raw Dataset Artifacts]
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Pure Deterministic Metrics Engine (Layer1MetricResult)                     │
│  - Mathematical completeness, duplicate rates, IQR outliers, Shannon entropy        │
│  - Regex PII scans, schema complexity index                                         │
│  - Immutable ground truth: score baseline (0.0 – 10.0)                              │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: Specialized Domain Expert Agents (DomainAgentVote)                         │
│  - Data Quality Analyst: Imputation strategies, outlier impact                      │
│  - Governance Guardian: GDPR/HIPAA risk, tokenization plans                         │
│  - Fairness Expert: Representation parity, cohort balance                           │
│  - Schema Expert: Key integrity, typing conformance                                 │
│  - Business Mapping: Table role awareness ('primary' vs 'reference')               │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: Consensus Judge with Mathematical Anchoring (ConsensusVerdict)             │
│  - Deterministic Anchoring: Clamps LLM judge within ±2.5 of Layer 1 baseline        │
│  - Severity-Based Disposition Escalation: READY < CONDITIONAL < SUSPENDED            │
│  - Deterministic Fatal Flaw Overrides:                                              │
│      * Critical PII (SSN, CC)  ──> Score capped at 2.0, SUSPENDED                   │
│      * Target nulls > 5%       ──> Score capped at 3.0, FAILED (primary tables only)│
│      * Schema complexity < 0.5 ──> Score capped at 7.9, CONDITIONAL                 │
│      * Zero applied fixes      ──> Score restricted to baseline + 0.5               │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
           [ push_to_target ]     [ await_approval ]       [ end_failed ]
             (Score >= 8.0,         (Score 6.0–7.9,         (Score < 6.0 or
              Disposition: READY)    Disposition: COND)      Disposition: SUSPENDED)
```

---

## Mathematical Formulations

### 1. Deterministic Score Anchoring ($\pm 2.5$ Band)
Let $S_{\text{base}} \in [0, 10]$ be the Layer 1 mathematical baseline, $S_{\text{raw}}$ be the qualitative score proposed by an LLM or agent, and $\beta = 2.5$ be the anchor band:

$$S_{\text{anchored}} = \max\Big(\max(0, S_{\text{base}} - \beta), \min(\min(10, S_{\text{base}} + \beta), S_{\text{raw}})\Big)$$

The LLM adds contextual nuance but is mathematically prevented from deviating by more than $\pm 2.5$ points from the empirical data facts.

### 2. Graded Breach Capping
To avoid artificial cliffs where mild and catastrophic failures cluster at the same floor, the score cap scales proportionally with breach severity $\sigma \in [0, 1]$:

$$C_{\text{ceiling}} = \max\Big(C_{\text{floor}}, \min(C_{\text{floor}} + 2.5, T_{\text{pass}} - 2.0)\Big)$$

$$S_{\text{capped}} = C_{\text{ceiling}} - (C_{\text{ceiling}} - C_{\text{floor}}) \times \sigma$$

---

## Installation

```bash
# Core schemas and anchoring engine
pip install langgraph-consensus-guard

# With LangGraph integration
pip install "langgraph-consensus-guard[langgraph]"
```

---

## Core Schema Contracts

### `Layer1MetricResult`
The immutable mathematical ground truth from Layer 1:
```python
from langgraph_consensus_guard import Layer1MetricResult

metric_result = Layer1MetricResult(
    run_id="run_001",
    dataset_ref="gold.orders",
    overall_score=8.2,
    passed=True,
    detected_pii=[{"column": "email", "type": "EMAIL"}],
    null_rates={"order_id": 0.0, "total_amount": 0.02},
    duplicate_rate=0.001,
    schema_complexity=0.85,
    total_rows=50000,
)
```

### `DomainAgentVote`
Structured qualitative assessment from Layer 2 domain agents:
```python
from langgraph_consensus_guard import DomainAgentVote, Disposition, ActionableFix

vote = DomainAgentVote(
    agent_id="governance_guardian",
    domain="governance",
    disposition=Disposition.CONDITIONAL,
    score=7.8,
    confidence=0.95,
    summary="Unmasked emails detected; requires tokenization before promotion.",
    findings=["Raw email addresses detected in metadata."],
    suggested_fixes=[
        ActionableFix(
            type="governance",
            addresses_check="pii_detection:email",
            description="Apply SHA-256 pseudonymization to email column.",
            auto_apply=False,
            risk_level="Medium",
        )
    ],
)
```

### `ConsensusVerdict`
Final synthesized outcome from Layer 3:
```python
from langgraph_consensus_guard import ConsensusVerdict, VerdictStatus

verdict = ConsensusVerdict(
    verdict=VerdictStatus.CONDITIONALLY_READY,
    global_readiness_score=7.8,
    deterministic_baseline_score=8.2,
    disposition=Disposition.CONDITIONAL,
    category_scores={"quality": 8.5, "governance": 7.8, "schema": 8.0},
    consensus_explanation="Anchored at 7.8. Governance guardian escalated disposition to CONDITIONAL.",
    overrides_applied=[],
    suggestions=[...],
    passed=False,
)
```

---

## Quickstart

```python
import asyncio
from langgraph_consensus_guard import (
    evaluate_records,
    DomainAgentVote,
    Disposition,
    ConsensusGuardState,
    build_consensus_guard_graph,
)

# 1. Sample dataset
records = [
    {"user_id": "U1", "age": 25, "income": 50000.0, "churn": 0},
    {"user_id": "U2", "age": 32, "income": None, "churn": 0},
    {"user_id": "U3", "age": 45, "income": 95000.0, "churn": 1},
]

# 2. Evaluate Layer 1 baseline mathematically
layer1 = evaluate_records(records, dataset_ref="users", run_id="run_101")

# 3. Simulate Layer 2 domain agent vote
votes = {
    "quality_agent": DomainAgentVote(
        agent_id="quality_agent",
        domain="quality",
        disposition=Disposition.READY,
        score=8.8,
        summary="Income null rate is within imputation tolerance.",
    )
}

# 4. Assemble state and execute ConsensusGuardGraph
async def run():
    state: ConsensusGuardState = {
        "run_id": "run_101",
        "dataset_ref": "users",
        "usecase": "Churn model training",
        "table_role": "primary",
        "target_column": "churn",
        "iteration": 1,
        "max_iterations": 3,
        "passing_threshold": 8.0,
        "layer1_metrics": layer1,
        "domain_votes": votes,
        "consensus_verdict": None,
        "agent_dialogue": [],
        "overrides_applied": [],
        "suggestions": [],
        "applied_fixes_history": ["init"],
        "status": "pending",
        "error": None,
    }

    graph = build_consensus_guard_graph()
    final_state = await graph.ainvoke(state)
    verdict = final_state["consensus_verdict"]

    print(f"Verdict: {verdict.verdict.value}")
    print(f"Global Score: {verdict.global_readiness_score:.2f} (Baseline: {verdict.deterministic_baseline_score:.2f})")
    print(f"Disposition: {verdict.disposition.value.upper()}")
    print(f"Cleared for Production: {verdict.passed}")

asyncio.run(run())
```

---

## Fatal Flaw Deterministic Overrides

| Rule | Condition | Scope | Enforced Cap | Resulting Disposition |
| :--- | :--- | :--- | :--- | :--- |
| **Critical PII Leakage** | SSN, Credit Card, or Passport detected | All tables | **Score $\le 2.0$** | `SUSPENDED` |
| **Primary Target Nulls** | Target column completeness $< 95\%$ | Primary tables only | **Score $\le 3.0$** | `SUSPENDED` |
| **Schema Simplicity** | Structural complexity $< 0.5$ | All tables | **Score $\le 7.9$** | `CONDITIONAL` |
| **Low Business Confidence**| Mapping confidence $< 0.5$ or poor data fitness | Primary tables only | **Score $\le 5.0$** | `SUSPENDED` |
| **ML Row Starvation** | Total rows $< 100$ for ML forecasting tasks | Primary tables only | **Score $\le 4.0$** | `CONDITIONAL` |
| **No-Op Relaxation** | Applied fix count $= 0$ | All runs | **Score $\le \text{Base} + 0.5$** | Preserved |

---

## Development & Quality Assurance

```bash
# Run strict type checking (Pyright)
npx --yes pyright

# Run linter & formatter checks (Ruff)
ruff check .

# Run complete unit test suite (Pytest)
python -m pytest -v

# Run end-to-end runnable demonstration
python examples/end_to_end_demo.py
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
