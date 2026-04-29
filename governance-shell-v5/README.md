# V5 Governance Shell

Date: 2026-04-29

## Purpose

V5 connects the relation-internalization diagnostic to a governance shell:

```text
review output -> approval gate -> decision log -> replay record -> responsibility chain
```

It does not add real engineering authority.

It asks:

```text
Can a bounded review workflow preserve relation-chain evidence, takeover
conditions, approval gates, replayability, and responsibility boundaries?
```

## Boundary

V5 is not:

```text
deployment governance
real construction approval
real safety assurance
legal responsibility automation
production workflow software
```

It is a toy diagnostic for auditability and takeover governance.

## Run

```bash
pip install -r requirements.txt
pytest -q
python -m src.run_v5_governance --config configs/v5_governance.yaml
python -m src.visualize_v5 --summary results/v5_governance_summary.csv
```

## Expected Outputs

```text
results/v5_governance_summary.csv
results/v5_governance_records.csv
results/v5_decision_log.json
reports/V5_GOVERNANCE_REPORT.md
reports/V5_SELF_AUDIT.md
reports/V5_CLAIMS.md
reports/V5_LIMITATIONS.md
figures/v5_gated_score.png
figures/v5_gate_enforcement.png
```

## V5.1 Hardening

Run:

```bash
python -m src.run_v51_hardening --config configs/v51_hardening.yaml
python -m src.visualize_v51 --summary results/v51_hardening_summary.csv
pytest -q
```

V5.1 attacks:

```text
fake replay hash
approval gate label without enforcement
responsibility boilerplate without human route
missing relation evidence in logs
route tampering after log creation
```

Read:

```text
reports/V5_FINAL_REPORT.md
```
