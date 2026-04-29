# V4 Engineering Review Case

Date: 2026-04-28

## Purpose

V4 is a small engineering-review-style diagnostic built on the V1-V3 mainline.

It asks:

```text
Can a system review a bounded slope-engineering plan by exposing the relevant
relation chain, uncertain links, verification points, takeover conditions, and
responsibility boundary?
```

## Boundary

V4 is not a real geotechnical safety model.

It does not predict real slope failure, approve real construction plans, or
replace expert review.

It is a diagnostic for whether an AI review output is:

```text
relation-chain specific
uncertainty-aware
auditable
human-takeover oriented
bounded by explicit failure conditions
```

## Stage Files

```text
V4_SCOPE.md
V4_CASE_SCHEMA.md
V4_REVIEW_PROTOCOL.md
V4_TASK_SPEC.md
```

## Current Status

```text
V4/V4.1/V4.2 implemented and stage-frozen
```

Run:

```bash
pip install -r requirements.txt
python -m src.run_v4_review --config configs/v4_review.yaml
python -m src.visualize_v4 --summary results/v4_summary.csv
pytest -q
```

Expected outputs:

```text
results/v4_summary.csv
results/v4_records.csv
figures/v4_review_scores.png
figures/v4_negative_control_rejection.png
reports/V4_ENGINEERING_REVIEW_REPORT.md
reports/V4_SELF_AUDIT.md
reports/V4_CLAIMS.md
reports/V4_LIMITATIONS.md
reports/V4_FINAL_REPORT.md
```

Run V4.1 hardening:

```bash
python -m src.run_v41_hardening --config configs/v41_hardening.yaml
python -m src.visualize_v41_hardening --summary results/v41_hardening_summary.csv
pytest -q
```

V4.1 attacks:

```text
schema-template shortcut
fluent but non-specific review text
case-order memorization
responsibility-boundary boilerplate
unsafe approval phrasing
```

Run V4.2 adversarial mutation:

```bash
python -m src.run_v42_mutation --config configs/v42_mutation.yaml
python -m src.visualize_v42_mutation --summary results/v42_mutation_summary.csv
pytest -q
```

V4.2 mutations:

```text
field order changed
irrelevant variables added
relation names paraphrased
hidden unsafe approval phrase inserted
```

## Do Not Claim

```text
real slope safety prediction
real monitoring system validity
deployment-ready engineering review
unrestricted temporal relation discovery
automatic plan approval
```
