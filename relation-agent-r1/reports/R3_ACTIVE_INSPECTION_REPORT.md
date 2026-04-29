# R3 Active Inspection Selection Report

## 1. Motivation

R3 tests whether an agent can choose the most informative field to inspect under multi-field missingness, noise, cost, and limited budget.

## 2. Tests

- Multiple critical and noncritical fields may be unknown at the same time.
- The agent must select an inspection target.
- The agent may inspect one field, update the observation, then decide again.
- Inspection value is rule-based: chain position plus action relevance plus conflict resolution minus noncritical penalty.
- Overinspection and unsafe automation are penalized under budget.

## 3. Results

| agent | n | inspection_target_accuracy | information_gain_efficiency | budgeted_safe_action_rate | unsafe_automation_rate | overinspection_rate | sequential_update_accuracy | cost_adjusted_success | r3_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_inspect_field | 5 | 0.000 | 0.140 | 0.000 | 0.000 | 1.000 | 0.920 | -0.800 | 0.000 |
| first_missing_inspect | 5 | 0.000 | -0.160 | 0.000 | 0.000 | 1.000 | 0.920 | -0.800 | 0.000 |
| missing_always_inspect | 5 | 0.000 | -0.160 | 0.000 | 0.000 | 1.000 | 0.920 | -0.800 | 0.000 |
| risk_first_inspect | 5 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.920 | -0.800 | 0.000 |
| active_inspection_agent | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.840 | 0.690 | 0.933 |

## 4. Baseline Margins

`first_missing_inspect` cost_adjusted_success: -0.800.
`random_inspect_field` cost_adjusted_success: -0.800.

## 5. Interpretation

A passing R3 agent should inspect the field with highest expected decision value, not merely inspect the first or any missing field.
