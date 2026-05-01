# Relation Internalization Program

This folder consolidates the main runnable project line.

## Main Claim

```text
Relation internalization requires usable internal relations:
transferable, counterfactual, editable, auditable, and action-guiding.
```

## Projects

```text
relation-agent-r1
```

R1 core non-LLM relation-internalization agent. It actively intervenes in a
small process world, learns an internal relation graph, uses that graph for
action, answers counterfactuals, supports relation editing, and is tested
against shortcut and passive-memory baselines. This is the current core agent
line.

```text
llm-relation-diagnostic
```

Sidecar black-box LLM diagnostic. It adapts the relation-internalization gates
to random-symbol prompts, support-conditioned binding, local edits, audit
specificity, missing-observation uncertainty, and budgeted inspect selection.
The default run uses mock baselines and does not claim LLM relation
understanding.

```text
prelinguistic-operational-structure-test
```

B-line PLOS-Test diagnostic. It tests whether operational structure can emerge
from non-linguistic continuous 2D dynamics along W -> O1 -> O2, using behavior,
structural intervention, and OOD gates. It deliberately excludes language labels
and does not treat prediction accuracy as evidence of operational structure.

```text
relation-internalization-test
```

V1 food-world explicit relation-table diagnostic.

```text
neural-relation-probe
```

Neural hidden-state probe, causal subspace intervention, and neural-to-table extraction.

```text
slope-relation-toy
```

Static slope-engineering relation-chain review toy.

```text
temporal-slope-relation-toy
```

V2/V2.1 delayed temporal relation-chain diagnostic and reviewer hardening. V3 adds partial observability, delayed noisy observations, takeover, unsafe automation, and uncertain relation audit.

```text
engineering-review-case-v4
```

V4/V4.1/V4.2 bounded engineering-review-case diagnostic and hardening. It tests review outputs for relation-chain specificity, uncertainty audit, takeover conditions, verification indicators, responsibility boundaries, schema-template shortcuts, fluent non-specific prose, case-order memory, boilerplate boundaries, unsafe approval phrasing, adversarial case mutations, irrelevant variables, relation-name paraphrases, and hidden unsafe approval phrases. It is not a real engineering approval system.

Read the frozen V4 stage report:

```text
engineering-review-case-v4\reports\V4_FINAL_REPORT.md
```

```text
governance-shell-v5
```

V5 toy governance shell for relation-chain review logs, approval gates, replay
records, takeover routing, responsibility traces, and non-deployment boundaries.
It is not real deployment governance.

Read the frozen V5 stage report:

```text
governance-shell-v5\reports\V5_FINAL_REPORT.md
```

```text
multi-party-audit-v6
```

V6 toy multi-party audit resolution diagnostic. It tests whether conflicting
review outputs are preserved, compared by relation evidence, routed to human
resolution, and protected against majority vote, confidence-only selection,
automatic compromise, minority-risk erasure, and missing audit trails.

Read the frozen V6 stage report:

```text
multi-party-audit-v6\reports\V6_FINAL_REPORT.md
```

## Summary Report

Read:

```text
relation-agent-r1\reports\R1_AGENT_REPORT.md
relation-agent-r1\reports\R1_SELF_AUDIT.md
relation-agent-r1\reports\R1_CLAIMS.md
relation-agent-r1\reports\R1_1_HARDENING_REPORT.md
relation-agent-r1\reports\R1_2_DISCOVERY_REPORT.md
relation-agent-r1\reports\R2_PARTIAL_OBSERVABILITY_REPORT.md
llm-relation-diagnostic\reports\LLM_RELATION_DIAGNOSTIC_REPORT.md
llm-relation-diagnostic\reports\LLM_RELATION_DIAGNOSTIC_LIVE_SMOKE_REPORT.md
llm-relation-diagnostic\reports\LLM_RELATION_DIAGNOSTIC_FREEZE_MEMO.md
llm-relation-diagnostic\reports\BUDGETED_INSPECT_STRESS_LIVE_REPORT.md
llm-relation-diagnostic\reports\BUDGETED_INSPECT_STRESS_MATRIX_REPORT.md
llm-relation-diagnostic\reports\LOCAL_EDIT_BEHAVIOR_STRESS_LIVE_REPORT.md
llm-relation-diagnostic\reports\AUDIT_CORRECTNESS_STRESS_LIVE_REPORT.md
prelinguistic-operational-structure-test\reports\B_LINE_RESEARCH_PROGRAM.md
prelinguistic-operational-structure-test\reports\B_LINE_PLOS_REPORT.md
prelinguistic-operational-structure-test\reports\B_LINE_SELF_AUDIT.md
prelinguistic-operational-structure-test\reports\B_LINE_SUBSTRATE_AUDIT.md
prelinguistic-operational-structure-test\reports\B_LINE_SUBSTRATE_SEARCH.md
prelinguistic-operational-structure-test\reports\B_LINE_FLOW_CHECKPOINT_HARDENING.md
prelinguistic-operational-structure-test\reports\B_LINE_EVIDENCE_LADDER.md
prelinguistic-operational-structure-test\reports\B2_1_TRACE_HARDENING_REPORT.md
prelinguistic-operational-structure-test\reports\B2_1A_TRACE_DEGENERACY_AUDIT.md
prelinguistic-operational-structure-test\reports\B2_2_TRACE_SELECTOR_DISENTANGLEMENT.md
prelinguistic-operational-structure-test\reports\B2_3_PRIVATE_TRACE_SELECTOR_REPORT.md
prelinguistic-operational-structure-test\reports\B3_DELAYED_TRACE_GUIDED_ACTIVE_INSPECTION.md
prelinguistic-operational-structure-test\reports\B3_1_ACTIVE_INSPECTION_DEGENERACY_AUDIT.md
LLM_RELATION_DIAGNOSTIC_SCOUTING.md
RELATION_INTERNALIZATION_PROJECT_SUMMARY.md
RELATION_INTERNALIZATION_V1_V5_FINAL_REPORT.md
RELATION_INTERNALIZATION_V1_V6_FINAL_REPORT.md
```

## Mainline Control Documents

Read these before extending the project:

```text
LONG_TERM_ROADMAP.md
MAINLINE_EXECUTION_PROTOCOL.md
CURRENT_SPRINT.md
PROJECT_EVOLUTION.md
CLAIM_BOUNDARY.md
NEXT_MAINLINE_TASKS.md
```

`LONG_TERM_ROADMAP.md` defines the corrected stage ladder. `MAINLINE_EXECUTION_PROTOCOL.md` defines how work proceeds without drifting. `CURRENT_SPRINT.md` records the active stage. `PROJECT_EVOLUTION.md` records how the broad discussion narrowed into the runnable mainline and why R1 returns to the core agent line. `CLAIM_BOUNDARY.md` defines what can and cannot be claimed. `NEXT_MAINLINE_TASKS.md` tracks the R2.1 route.

## Verification Commands

Run each project independently:

```bash
cd relation-agent-r1
pytest -q
python -m src.run_r1_experiment --config configs/r1.yaml
python -m src.visualize_r1 --summary results/r1_summary.csv
python -m src.run_r11_hardening --config configs/r11_hardening.yaml
python -m src.visualize_r11 --summary results/r11_hardening_summary.csv
python -m src.run_r12_discovery --config configs/r12_discovery.yaml
python -m src.visualize_r12 --summary results/r12_discovery_summary.csv
python -m src.run_r2_partial_observability --config configs/r2_partial_observability.yaml
python -m src.visualize_r2 --summary results/r2_partial_observability_summary.csv

cd ..\llm-relation-diagnostic
pytest -q
python -m src.run_experiment --config configs/base.yaml
python -m src.run_experiment --config configs/strict.yaml
python -m src.run_experiment --config configs/budgeted_inspect_stress.yaml
python -m src.run_experiment --config configs/local_edit_behavior_stress.yaml
python -m src.run_experiment --config configs/audit_correctness_stress.yaml

# Optional live LLM run, after starting a llama.cpp server separately:
python -m src.run_experiment --config configs/base.yaml --solvers llama_cpp --base-url http://127.0.0.1:8083 --model qwen2.5-3b-instruct-q5_k_m --label qwen3b_smoke

# Optional local GGUF matrix for the budgeted-inspect stress set:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_matrix.ps1 -Config configs\budgeted_inspect_stress.yaml -Models qwen05b,qwen15b,qwen3b,gemma3_4b

cd ..\prelinguistic-operational-structure-test
pytest -q
python -m src.run_sweep --config configs/sweep.yaml
python -m src.run_hardening --config configs/sweep.yaml --seed 0
python -m src.run_b11_flow_hardening --config configs/b11_flow_hardening.yaml --seed 0
python -m src.run_b2_delayed_checkpoint --config configs/b2_delayed_checkpoint.yaml --seed 0
python -m src.run_b21_trace_hardening --config configs/b21_trace_hardening.yaml --seed 0
python -m src.run_b21a_degeneracy_audit --config configs/b21a_degeneracy_audit.yaml --seed 0
python -m src.run_b22_selector_disentanglement --config configs/b22_selector_disentanglement.yaml --seed 0
python -m src.run_b23_private_selector --config configs/b23_private_selector.yaml --seed 0
python -m src.run_b3_active_inspection --config configs/b3_active_inspection.yaml --seed 0
python -m src.run_b31_inspection_audit --config configs/b31_inspection_audit.yaml --seed 0
python -m src.run_b32_mechanism_inspection --config configs/b32_mechanism_inspection.yaml --seed 0
python -m src.run_b4_intervention --config configs/b4_intervention.yaml --seed 0
python -m src.run_b41_intervention_audit --config configs/b41_intervention_audit.yaml --seed 0
python -m src.run_b42_action_type_disambiguation --config configs/b42_action_type_disambiguation.yaml --seed 0
python -m src.run_b5_closed_loop --config configs/b5_closed_loop.yaml --seed 0
python -m src.run_b51_closed_loop_audit --config configs/b51_closed_loop_audit.yaml --seed 0
python -m src.run_b5_clean_closed_loop --config configs/b5_clean_closed_loop.yaml --seed 0
python -m src.run_b51_clean_closed_loop_audit --config configs/b51_clean_closed_loop_audit.yaml --seed 0
python -m src.run_b52_adaptive_update --config configs/b52_adaptive_update.yaml --seed 0
python -m src.run_b6_risk_constrained_loop --config configs/b6_risk_constrained_loop.yaml --seed 0
python -m src.visualize_b22 --summary results/b22_selector_disentanglement_summary.csv
python -m src.visualize_b23 --summary results/b23_private_selector_summary.csv
python -m src.visualize_b3 --summary results/b3_active_inspection_summary.csv
python -m src.visualize_b31 --summary results/b31_inspection_audit_summary.csv
python -m src.visualize_b32 --summary results/b32_mechanism_inspection_summary.csv
python -m src.visualize_b4 --summary results/b4_intervention_summary.csv
python -m src.visualize_b41 --summary results/b41_intervention_audit_summary.csv
python -m src.visualize_b42 --summary results/b42_action_type_summary.csv
python -m src.visualize_b5 --summary results/b5_closed_loop_summary.csv
python -m src.visualize_b51 --summary results/b51_closed_loop_audit_summary.csv
python -m src.visualize_b5_clean --summary results/b5_clean_closed_loop_summary.csv --audit results/b51_clean_closed_loop_audit_summary.csv
python -m src.visualize_b52 --summary results/b52_adaptive_update_summary.csv
python -m src.visualize_b6 --summary results/b6_risk_constrained_summary.csv
python -m src.visualize --summary results/overall_summary.csv

cd ..\relation-internalization-test
pytest -q

cd ..\neural-relation-probe
pytest -q

cd ..\slope-relation-toy
pytest -q

cd ..\temporal-slope-relation-toy
pytest -q
python -m src.run_hardening --config configs/hardening.yaml
python -m src.run_v3_uncertainty --config configs/v3_uncertainty.yaml
python -m src.visualize_v3_uncertainty --summary results/v3_uncertainty_summary.csv
```

Run V4:

```bash
cd engineering-review-case-v4
pytest -q
python -m src.run_v4_review --config configs/v4_review.yaml
python -m src.visualize_v4 --summary results/v4_summary.csv
python -m src.run_v41_hardening --config configs/v41_hardening.yaml
python -m src.visualize_v41_hardening --summary results/v41_hardening_summary.csv
python -m src.run_v42_mutation --config configs/v42_mutation.yaml
python -m src.visualize_v42_mutation --summary results/v42_mutation_summary.csv

cd ..\governance-shell-v5
pytest -q
python -m src.run_v5_governance --config configs/v5_governance.yaml
python -m src.visualize_v5 --summary results/v5_governance_summary.csv
python -m src.run_v51_hardening --config configs/v51_hardening.yaml
python -m src.visualize_v51 --summary results/v51_hardening_summary.csv

cd ..\multi-party-audit-v6
pytest -q
python -m src.run_v6_audit --config configs/v6_audit.yaml
python -m src.visualize_v6 --summary results/v6_audit_summary.csv
python -m src.run_v61_hardening --config configs/v61_hardening.yaml
python -m src.visualize_v61 --summary results/v61_hardening_summary.csv
```

## Boundary

This is a research diagnostic program. It is not a real geotechnical safety model and makes no deployment claim.
