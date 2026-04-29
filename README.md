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
