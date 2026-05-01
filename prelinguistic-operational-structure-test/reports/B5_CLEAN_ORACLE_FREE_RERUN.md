# B5-Clean Oracle-Free Closed-Loop Rerun

## 1. Purpose

B5.1 found value/oracle leakage in B5 policy input. B5-Clean fixes the experimental hygiene by separating model_input, evaluator_ground_truth, and oracle_baseline_view.

## 2. Problem Found by B5.1

- value_leakage_count = 800.000
- scripted_update_score = 1.000
- model_gain_over_scripted_update = 0.000
- feedback_revision_over_scripted_ratio = 1.000
- cross_model_exact_plan_match_rate = 1.000

## 3. Clean Separation

- model_input: policy-visible only
- evaluator_ground_truth: metrics-only
- oracle_baseline_view: oracle baseline only

## 4. Clean B5 Results

| model | seed | clean_b5_closed_loop_score | original_b5_closed_loop_score | score_drop_from_original | inspect_timing_accuracy | epistemic_value_alignment | trace_update_accuracy | post_inspection_intervention_accuracy | pragmatic_value_alignment | feedback_revision_accuracy | planning_budget_compliance | clean_random_closed_loop_score | clean_oracle_closed_loop_score | model_input_leakage_count | policy_output_oracle_usage_rate | evaluator_ground_truth_policy_access_count | oracle_baseline_access_violation_count | clean_b5_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 0.925 | 0.887 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| field_memory_model | 0 | 0.925 | 0.887 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| schema_memory_model | 0 | 0.925 | 0.887 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## 5. Interpretation

B5-Clean fixes the oracle/value leakage found by B5.1. Under sanitized model inputs and evaluator-only ground truth, B5 remains a valid oracle-free closed-loop path diagnostic in the toy PLOS environment.

## 6. Claim Boundary

Do not claim real control.
Do not claim robotics ability.
Do not claim engineering deployment.
Do not claim human-like active inference.
Do not claim language-free cognition solved.

## Clean B5.1 Rerun Results

| model | seed | cross_model_exact_plan_match_rate | exact_all_model_same_plan_rate | fixed_closed_loop_plan_rate | inspect_always_rate | intervene_immediately_rate | always_inspect_then_intervene_rate | skip_inspect_when_not_needed_rate | skip_intervention_when_not_needed_rate | decision_diversity_score | shared_closed_loop_policy_usage_rate | private_trace_closed_loop_usage_rate | fallback_usage_rate | oracle_plan_usage_rate | oracle_trace_update_usage_rate | oracle_feedback_revision_usage_rate | trace_update_specificity | trace_update_ablation_drop | trace_update_over_non_trace_ratio | wrong_inspection_update_drop | shuffled_inspection_update_drop | scripted_update_score | model_gain_over_scripted_update | feedback_revision_specificity | feedback_revision_ablation_drop | feedback_revision_over_scripted_ratio | contradictory_feedback_sensitivity | random_closed_loop_score | saliency_closed_loop_score | short_horizon_closed_loop_score | inspect_always_score | intervene_immediately_score | oracle_closed_loop_score | model_gain_over_inspect_always | model_gain_over_intervene_immediately | value_leakage_count | planning_budget_stress_retention | strict_budget_compliance | b51_clean_closed_loop_audit_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 1.000 | 0.113 | 0.500 | 0.250 | 0.500 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.300 | 3.000 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 0.300 | 1.000 | 0.300 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 1.000 | 0.000 |
| field_memory_model | 0 | 1.000 | 1.000 | 0.113 | 0.500 | 0.250 | 0.500 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.300 | 3.000 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 0.300 | 1.000 | 0.300 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 1.000 | 0.000 |
| schema_memory_model | 0 | 1.000 | 1.000 | 0.113 | 0.500 | 0.250 | 0.500 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.300 | 3.000 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 0.300 | 1.000 | 0.300 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 1.000 | 0.000 |

Clean B5.1 reruns the degeneracy checks after model-input sanitization. Leakage is evaluated from the clean leakage audit records.
