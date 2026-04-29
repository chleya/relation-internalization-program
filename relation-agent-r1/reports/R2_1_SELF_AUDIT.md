# R2.1 Self-Audit

## What This Improves

- Separates relation-specific uncertainty from blanket inspection.
- Adds non-critical missingness controls.
- Adds inspection cost and a limited inspection budget.
- Requires conflict localization for named relation links.
- Makes inspection non-oracle by revealing only one selected field.

## Remaining Weaknesses

- This remains a toy diagnostic. It is not real slope monitoring, not calibrated inspection policy, not adversarial missingness robustness, and not deployment-ready engineering AI.
- Missingness, conflicts, and inspection noise are synthetic.
- The policy is hand-shaped for the diagnostic relation graph.
- Inspection still uses a simple field-selection interface.

## Current R2.1 Result

- relation_specific_uncertainty_agent r2_1_gated_score: 0.933
