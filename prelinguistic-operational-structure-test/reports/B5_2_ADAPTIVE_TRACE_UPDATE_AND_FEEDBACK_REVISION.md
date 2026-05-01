# B5.2 Adaptive Trace Update and Feedback Revision

## 1. Purpose

B5-Clean fixed oracle/value leakage, but clean B5.1 still failed due to same-plan degeneracy and scripted update / feedback explanations. B5.2 tests whether trace update and feedback revision actually depend on inspection and consequence content.

## 2. Background

PLOS v1 through B4.2 established the one-shot operational chain. B5 established a clean closed-loop path after B5-Clean. Clean B5.1 found remaining same-plan, scripted update, and scripted feedback failures.

## 3. Tests

- inspection-content swap
- counterfactual inspection observation
- same-initial-different-info plan divergence
- contradictory feedback
- delayed feedback
- update-vs-scripted baseline
- feedback-vs-scripted baseline
- revision-specific ablation

## 4. Results

| model | seed | value_leakage_count | inspection_content_sensitivity | inspection_swap_update_change_rate | counterfactual_update_switch_rate | counterfactual_plan_switch_rate | same_initial_different_info_plan_divergence | post_update_plan_divergence | post_update_intervention_change_rate | model_update_score | scripted_update_score | model_gain_over_scripted_update | update_specificity_over_scripted | feedback_content_sensitivity | contradictory_feedback_revision_accuracy | delayed_feedback_revision_accuracy | model_feedback_score | scripted_feedback_score | model_gain_over_scripted_feedback | feedback_specificity_over_scripted | revision_specific_ablation_drop | update_path_ablation_drop | feedback_path_ablation_drop | non_revision_path_stability | cross_model_exact_plan_match_rate | exact_all_model_same_plan_rate | oracle_adaptive_update_score | random_update_score | b52_adaptive_update_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.013 | 0.988 | 79.994 | 1.000 | 1.000 | 1.000 | 1.000 | 0.017 | 0.983 | 59.996 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.996 |
| field_memory_model | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.017 | 0.983 | 59.996 | 1.000 | 1.000 | 1.000 | 1.000 | 0.013 | 0.988 | 79.994 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.996 |
| schema_memory_model | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.021 | 0.979 | 47.998 | 1.000 | 1.000 | 1.000 | 1.000 | 0.025 | 0.975 | 39.998 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.993 |

## 5. Interpretation

B5.2 reduces the clean B5.1 concerns of same-plan degeneracy and scripted update/feedback under current toy diagnostics.

## 6. Claim Boundary

Do not claim real control.
Do not claim robotics ability.
Do not claim engineering deployment.
Do not claim human-like active inference.
Do not claim language-free cognition solved.
