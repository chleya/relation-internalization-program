# B3 Delayed Trace-Guided Active Inspection

## 1. Purpose

B3 tests whether B2.3 private delayed traces can guide active inspection under budget.

## 2. Background

PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing path solves delayed checkpoint. B2.1: trace hardening. B2.1a: identical-score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection.

## 3. Task

The model has limited inspection budget and must choose one region. Correct inspection should follow delayed trace, not saliency or short-horizon checkpoint.

## 4. Models

- recurrent_flow_checkpoint_model
- field_memory_model
- schema_memory_model

## 5. Baselines

- random inspection
- saliency inspection
- short-horizon checkpoint inspection
- oracle inspection

## 6. Metrics

- trace-guided inspection accuracy
- trace-vs-saliency rejection
- delayed information gain
- gain over random
- gain over saliency
- gain over short-horizon
- trace ablation inspection drop
- delay OOD inspection

## 7. Results

| model | seed | trace_guided_inspection_accuracy | trace_vs_saliency_rejection | delayed_information_gain | relative_information_gain | inspection_value_gain_over_random | inspection_value_gain_over_saliency | inspection_value_gain_over_short_horizon | base_trace_guided_inspection_accuracy | trace_ablated_inspection_accuracy | trace_ablation_inspection_drop | delay_ood_inspection_accuracy | oracle_inspection_score | random_inspection_score | saliency_inspection_score | short_horizon_inspection_score | b3_active_inspection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.936 | 0.950 | 0.950 | 1.000 | 0.000 | 1.000 | 0.975 | 1.000 | 0.006 | 0.000 | 0.000 | 0.985 |
| field_memory_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.936 | 0.950 | 0.950 | 1.000 | 0.000 | 1.000 | 0.975 | 1.000 | 0.006 | 0.000 | 0.000 | 0.985 |
| schema_memory_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.936 | 0.950 | 0.950 | 1.000 | 0.000 | 1.000 | 0.975 | 1.000 | 0.006 | 0.000 | 0.000 | 0.985 |

## 8. Interpretation

Private delayed trace can guide budgeted active inspection in the toy PLOS world.

B3 does not re-audit mechanism separation. Identical B3 scores across the private-selector models mean that all three solve the current active-inspection task under the same inspection-value protocol; they are not additional evidence of fully independent mechanisms.

## 9. Claim Boundary

Do not claim general active intelligence.
Do not claim real-world robot inspection.
Do not claim human-like attention.
Do not claim language-free cognition solved.
