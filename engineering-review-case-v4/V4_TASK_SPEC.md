# V4 Implementation Task Spec

Date: 2026-04-28

## Project Name

```text
engineering-review-case-v4
```

## Root

```text
F:\relation-internalization-program\engineering-review-case-v4
```

## Objective

Implement a small runnable V4 diagnostic that evaluates bounded slope-style
engineering review cases.

Do not use real engineering data.
Do not perform real slope safety prediction.
Do not add new theory.

## Required Directory Structure

```text
engineering-review-case-v4/
  README.md
  V4_SCOPE.md
  V4_CASE_SCHEMA.md
  V4_REVIEW_PROTOCOL.md
  V4_TASK_SPEC.md
  configs/
    v4_review.yaml
  cases/
    toy_cases.json
    negative_reviews.json
  src/
    __init__.py
    case_schema.py
    reviewers.py
    metrics_v4.py
    run_v4_review.py
    visualize_v4.py
    report_v4.py
  tests/
    test_case_schema.py
    test_review_scoring.py
    test_negative_controls.py
    test_claim_boundary.py
  results/
    .gitkeep
  figures/
    .gitkeep
  reports/
    .gitkeep
```

## Required Reviewers

```text
generic_review
surface_warning_review
structural_memory_review
relation_chain_review
uncertainty_aware_review
```

## Required Metrics

```text
relation_chain_specificity
action_point_mapping
uncertainty_takeover_quality
verification_indicator_quality
responsibility_boundary_quality
unsafe_review_rejection
gated_v4_score
```

## Required Commands

```bash
python -m src.run_v4_review --config configs/v4_review.yaml
python -m src.visualize_v4 --summary results/v4_summary.csv
pytest -q
```

## Required Outputs

```text
results/v4_summary.csv
figures/v4_review_scores.png
figures/v4_negative_control_rejection.png
reports/V4_ENGINEERING_REVIEW_REPORT.md
reports/V4_SELF_AUDIT.md
reports/V4_CLAIMS.md
reports/V4_LIMITATIONS.md
```

## Failure Cases To Report

Report explicitly if:

```text
generic_review passes gated_v4_score
surface_warning_review passes false-warning cases
structural_memory_review passes without uncertainty/takeover audit
uncertainty_aware_review approves any case as safe
reviews reward polished prose more than relation-chain specificity
case schema starts resembling real engineering approval
```

## Best Expected Result

```text
uncertainty_aware_review passes V4 gates
relation_chain_review passes most relation/action gates but may fail uncertainty depth
generic_review, surface_warning_review, and structural_memory_review have gated_v4_score = 0.0
```

