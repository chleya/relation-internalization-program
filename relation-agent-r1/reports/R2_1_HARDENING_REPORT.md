# R2.1 Reviewer Hardening Report

## 1. Motivation

R2.1 tests whether inspection is relation-specific rather than a blanket response to any unknown field.

## 2. Tests

- Non-critical missingness should not force inspection.
- Critical missingness should trigger inspection when the relation chain is unverifiable.
- Mixed observability should allow action when an observed downstream path is sufficient.
- Conflict audits must localize relation links.
- Inspection has cost and a limited budget.
- Inspection reveals one selected field, not the full true state.
- The `missing_always_inspect` baseline is compared under cost-adjusted success and inspection precision.

## 3. Results

| agent | n | critical_missing_inspection_recall | noncritical_missing_no_inspect_rate | inspection_precision | unnecessary_inspection_rate | unsafe_automation_rate | conflict_localization_accuracy | cost_adjusted_success | non_oracle_inspection_update_accuracy | r2_1_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| missing_always_inspect | 5 | 1.000 | 0.000 | 0.333 | 1.000 | 0.375 | 0.000 | -0.522 | 0.915 | 0.000 |
| relation_specific_uncertainty_agent | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.548 | 0.915 | 0.933 |

## 4. Baseline Margin

`missing_always_inspect` cost_adjusted_success: -0.522. A passing agent must exceed this by at least 0.100.

## 5. Interpretation

A passing R2.1 agent inspects because a specific physical relation link cannot be verified, not because any field is unknown. The hardening target is inspection precision under uncertainty, not maximal conservatism.

## 6. Boundary

R2.1 remains a toy diagnostic. It is not real slope monitoring, not calibrated inspection policy, not adversarial missingness robustness, and not deployment-ready engineering AI.
