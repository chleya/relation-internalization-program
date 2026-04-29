# R1 Relation Internalization Agent

Date: 2026-04-29

## Purpose

R1 returns to the core line:

```text
Build a non-LLM agent that learns usable internal relations through interaction.
```

This is not a review or governance shell. It is a minimal agent-body experiment.

## Core Question

```text
Can an agent actively intervene in a small process world, learn an internal
relation graph, use it for action, answer counterfactuals, and change behavior
after relation edits?
```

## Boundary

R1 is still a toy diagnostic.

It does not claim:

```text
real engineering intelligence
LLM replacement
general world modeling
real slope safety ability
deployment readiness
```

## Run

```bash
pip install -r requirements.txt
pytest -q
python -m src.run_r1_experiment --config configs/r1.yaml
python -m src.visualize_r1 --summary results/r1_summary.csv
python -m src.run_r11_hardening --config configs/r11_hardening.yaml
python -m src.visualize_r11 --summary results/r11_hardening_summary.csv
python -m src.run_r12_discovery --config configs/r12_discovery.yaml
python -m src.visualize_r12 --summary results/r12_discovery_summary.csv
python -m src.run_r2_partial_observability --config configs/r2_partial_observability.yaml
python -m src.visualize_r2 --summary results/r2_partial_observability_summary.csv
python -m src.run_r2_1_hardening --config configs/r2_1_hardening.yaml
python -m src.visualize_r2_1 --summary results/r2_1_summary.csv
python -m src.run_r3_active_inspection --config configs/r3_active_inspection.yaml
python -m src.visualize_r3 --summary results/r3_summary.csv
```

## Expected Outputs

```text
results/r1_summary.csv
results/r11_hardening_summary.csv
results/r12_discovery_summary.csv
results/r2_partial_observability_summary.csv
results/r2_1_summary.csv
results/r2_1_records.csv
results/r3_summary.csv
results/r3_records.csv
results/r1_relations.json
reports/R1_AGENT_REPORT.md
reports/R1_1_HARDENING_REPORT.md
reports/R1_1_SELF_AUDIT.md
reports/R1_2_DISCOVERY_REPORT.md
reports/R1_2_SELF_AUDIT.md
reports/R2_STAGE_REPORT.md
reports/R2_PARTIAL_OBSERVABILITY_REPORT.md
reports/R2_SELF_AUDIT.md
reports/R2_1_HARDENING_REPORT.md
reports/R2_1_SELF_AUDIT.md
reports/R3_ACTIVE_INSPECTION_REPORT.md
reports/R3_SELF_AUDIT.md
reports/R1_SELF_AUDIT.md
reports/R1_CLAIMS.md
reports/R1_LIMITATIONS.md
figures/r1_scores.png
figures/r1_relation_recovery.png
figures/r11_hardening_score.png
figures/r12_discovery_score.png
figures/r2_partial_observability_score.png
figures/r2_1_gate_scores.png
figures/inspection_precision_recall.png
figures/unsafe_vs_unnecessary.png
figures/cost_adjusted_success.png
figures/r3_gate_scores.png
figures/inspection_target_accuracy.png
figures/information_gain_efficiency.png
figures/budgeted_safe_action_rate.png
figures/unsafe_vs_overinspection.png
```
