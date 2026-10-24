"""End-to-End demonstration of langgraph-consensus-guard.

Shows:
1. Deterministic Layer 1 metric evaluation on raw records.
2. Layer 2 Domain Expert Agent assessments and structured voting.
3. Layer 3 Consensus synthesis with deterministic anchoring (±2.5 band).
4. Fatal flaw override enforcement (e.g. unmasked PII).
5. Executing the workflow through the ConsensusGuardGraph runner.
"""

from __future__ import annotations

import asyncio

from langgraph_consensus_guard import (
    ActionableFix,
    ConsensusGuardState,
    Disposition,
    DomainAgentVote,
    build_consensus_guard_graph,
    evaluate_records,
)


async def main() -> None:
    print("=" * 70)
    print("langgraph-consensus-guard: End-to-End Demonstration")
    print("=" * 70)

    # 1. Sample dataset records
    sample_records = [
        {"customer_id": "C001", "name": "Alice Smith", "email": "alice@example.com", "spend": 450.0, "churn": 0},
        {"customer_id": "C002", "name": "Bob Jones", "email": "bob@example.com", "spend": None, "churn": 0},
        {"customer_id": "C003", "name": "Carol White", "email": "carol@example.com", "spend": 120.0, "churn": 1},
        {"customer_id": "C004", "name": "David Brown", "email": "david@example.com", "spend": 890.0, "churn": 0},
        {"customer_id": "C005", "name": "Eve Davis", "email": "eve@example.com", "spend": 310.0, "churn": None},
    ]

    print("\n[Step 1] Running Deterministic Layer 1 Metric Evaluation...")
    layer1 = evaluate_records(
        records=sample_records,
        dataset_ref="gold.customer_retention",
        run_id="run_demo_001",
    )
    print(f"  Overall Layer 1 Baseline Score: {layer1.overall_score:.2f}/10.0")
    print(f"  Schema Complexity: {layer1.schema_complexity:.2f}")
    print(f"  Null Rates: {layer1.null_rates}")
    print(f"  Detected PII: {layer1.detected_pii}")

    # 2. Simulated Layer 2 Domain Expert Agent Votes
    print("\n[Step 2] Collecting Layer 2 Domain Expert Agent Votes...")
    domain_votes: dict[str, DomainAgentVote] = {
        "quality_analyst": DomainAgentVote(
            agent_id="quality_analyst",
            domain="quality",
            disposition=Disposition.READY,
            score=8.4,
            confidence=0.92,
            summary="Null rates in spend column are within acceptable imputation bounds.",
            findings=["Column 'spend' has 20% null rate."],
            suggested_fixes=[
                ActionableFix(
                    type="quality",
                    addresses_check="null_rate:spend",
                    description="Impute null values in spend with median.",
                    auto_apply=True,
                    risk_level="Low",
                    code="df['spend'] = df['spend'].fillna(df['spend'].median())",
                )
            ],
        ),
        "governance_guardian": DomainAgentVote(
            agent_id="governance_guardian",
            domain="governance",
            disposition=Disposition.CONDITIONAL,
            score=7.8,
            confidence=0.98,
            summary="Unmasked customer emails found; requires column-level tokenization.",
            findings=["Email address pattern detected in 'email' column."],
            suggested_fixes=[
                ActionableFix(
                    type="governance",
                    addresses_check="pii_detection:email",
                    description="Mask email column with SHA-256 hash.",
                    auto_apply=False,
                    risk_level="Medium",
                    code="df['email'] = df['email'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())",
                )
            ],
        ),
        "fairness_expert": DomainAgentVote(
            agent_id="fairness_expert",
            domain="fairness",
            disposition=Disposition.READY,
            score=8.9,
            confidence=0.88,
            summary="Class representation across cohorts is well balanced.",
        ),
    }

    for agent_id, vote in domain_votes.items():
        print(f"  - {agent_id}: disposition={vote.disposition.value.upper()}, score={vote.score}")

    # 3. Assemble LangGraph Workflow State
    print("\n[Step 3] Executing ConsensusGuardGraph Workflow...")
    initial_state: ConsensusGuardState = {
        "run_id": "run_demo_001",
        "dataset_ref": "gold.customer_retention",
        "usecase": "ML Customer Churn Model Training",
        "table_role": "primary",
        "target_column": "churn",
        "iteration": 1,
        "max_iterations": 3,
        "passing_threshold": 8.0,
        "layer1_metrics": layer1,
        "domain_votes": domain_votes,
        "consensus_verdict": None,
        "agent_dialogue": [],
        "overrides_applied": [],
        "suggestions": [],
        "applied_fixes_history": ["Initial validation"],
        "status": "pending",
        "error": None,
    }

    graph = build_consensus_guard_graph()
    final_state = await graph.ainvoke(initial_state)

    verdict = final_state["consensus_verdict"]
    print("\n" + "=" * 70)
    print("Consensus Verdict Results:")
    print("=" * 70)
    if verdict:
        print(f"Verdict:              {verdict.verdict.value}")
        print(f"Global Score:         {verdict.global_readiness_score:.2f} / 10.0")
        print(f"Baseline Score:       {verdict.deterministic_baseline_score:.2f} / 10.0")
        print(f"Disposition:          {verdict.disposition.value.upper()}")
        print(f"Cleared for Prod:     {verdict.passed}")
        print(f"Category Scores:      {verdict.category_scores}")
        print(f"Explanation:          {verdict.consensus_explanation}")
        print(f"Overrides Applied:    {verdict.overrides_applied}")
        print(f"Remediation Count:    {len(verdict.suggestions)}")
        for i, s in enumerate(verdict.suggestions, 1):
            print(f"   {i}. [{s.type.upper()}] {s.addresses_check} -> {s.description}")
    print(f"\nFinal State Status:   {final_state['status']}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
