# Temporal Slope Relation Toy

V2 for relation internalization:

```text
relations enter time
```

Core delayed chain:

```text
rainfall[t] -> pore_pressure[t+1]
pore_pressure[t] -> displacement[t+1]
displacement[t] + sparse monitoring[t] -> crack[t+1]
crack[t] -> risk[t]
```

Run:

```bash
pip install -r requirements.txt
python -m src.run_experiment --seeds 0 1 2 3 4
python -m src.visualize --summary results/temporal_summary.csv
python -m src.self_audit --summary results/temporal_summary.csv --output reports/temporal_self_audit.md
python -m src.run_hardening --config configs/hardening.yaml
python -m src.visualize_hardening --summary results/hardening_summary.csv
python -m src.run_v3_uncertainty --config configs/v3_uncertainty.yaml
python -m src.visualize_v3_uncertainty --summary results/v3_uncertainty_summary.csv
python -m src.run_v31_hardening --config configs/v31_hardening.yaml
python -m src.visualize_v31_hardening --summary results/v31_hardening_summary.csv
python -m src.run_v32_stress --config configs/v32_stress.yaml
python -m src.visualize_v32_stress --summary results/v32_stress_summary.csv
pytest -q
```

Outputs:

```text
results/temporal_summary.csv
figures/temporal_scores.png
reports/temporal_auto_report.md
reports/temporal_self_audit.md
reports/V2_TEMPORAL_REPORT.md
reports/V2_1_HARDENING_REPORT.md
reports/V2_1_SELF_AUDIT.md
reports/V2_1_FINAL_REPORT.md
reports/V2_1_LIMITATIONS.md
reports/V2_1_CLAIMS.md
reports/V3_CHARTER.md
results/v3_uncertainty_summary.csv
figures/v3_noisy_action_success.png
figures/v3_takeover_recall.png
figures/v3_unsafe_automation_rate.png
figures/v3_gated_score.png
reports/V3_UNCERTAINTY_REPORT.md
reports/V3_SELF_AUDIT.md
reports/V3_FINAL_REPORT.md
reports/V3_CLAIMS.md
reports/V3_LIMITATIONS.md
results/v31_hardening_summary.csv
figures/v31_takeover_overuse_control.png
figures/v31_conflicting_evidence_takeover.png
figures/v31_audit_specificity.png
figures/v31_hardening_gated_score.png
reports/V3_1_HARDENING_REPORT.md
reports/V3_1_SELF_AUDIT.md
reports/V3_1_CLAIMS.md
reports/V3_1_LIMITATIONS.md
results/v32_stress_summary.csv
figures/v32_long_missing_takeover_recall.png
figures/v32_correlated_failure_recall.png
figures/v32_multi_conflict_audit_score.png
figures/v32_stress_gated_score.png
reports/V3_2_STRESS_REPORT.md
reports/V3_2_SELF_AUDIT.md
reports/V3_2_CLAIMS.md
reports/V3_2_LIMITATIONS.md
```

V2.1 hardening checks:

```text
variable delay
false temporal shortcut
multi-link delay edit
temporal audit consistency
anti-template generalization
```

V2.1 is frozen. Next stage:

```text
V3: partial observability + missing sensors + delayed noisy observations + takeover threshold
```

V3 adds:

```text
missing sensors
delayed noisy observations
takeover action
takeover precision / recall
unsafe automation rate
uncertain relation audit
```

V3.1 hardening checks:

```text
irrelevant missing sensor rejection
benign noise rejection
takeover overuse control
conflicting evidence takeover
audit specificity
```

V3.2 stress checks:

```text
long missing spans
correlated sensor failures
drift-like conflicts
multi-conflict audit
delayed response safety
```
