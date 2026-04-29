# V6 Multi-Party Audit Resolution

Date: 2026-04-29

## Purpose

V6 tests what happens when two or more plausible review outputs disagree.

It asks:

```text
Can a toy audit resolver preserve disagreement, compare relation evidence,
avoid automatic resolution, route to human review, and keep responsibility
traceable?
```

## Boundary

V6 is not:

```text
real dispute resolution
legal adjudication
engineering approval
deployment governance
expert replacement
```

It is a toy diagnostic for multi-review audit behavior.

## Run

```bash
pip install -r requirements.txt
pytest -q
python -m src.run_v6_audit --config configs/v6_audit.yaml
python -m src.visualize_v6 --summary results/v6_audit_summary.csv
```

Run V6.1 hardening:

```bash
python -m src.run_v61_hardening --config configs/v61_hardening.yaml
python -m src.visualize_v61 --summary results/v61_hardening_summary.csv
pytest -q
```

## Expected Outputs

```text
results/v6_audit_summary.csv
results/v6_audit_records.csv
results/v6_resolution_log.json
reports/V6_AUDIT_REPORT.md
reports/V6_SELF_AUDIT.md
reports/V6_CLAIMS.md
reports/V6_LIMITATIONS.md
figures/v6_gated_score.png
figures/v6_disagreement_detection.png
figures/v6_no_auto_resolution.png
```

V6 final report:

```text
reports/V6_FINAL_REPORT.md
```
