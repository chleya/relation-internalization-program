# B5.1 Closed-Loop Degeneracy Audit

## 1. Purpose

B5 produced strong closed-loop results. B5.1 audits whether this reflects genuine private delayed operational trace closed-loop operation or fixed scripts, shortcut policies, oracle-like updates, value leakage, weak baselines, loose planning budget, or nonspecific ablations.

## 2. Background

PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: inspection degeneracy audit. B3.2: mechanism-disambiguating active inspection. B4: trace-guided intervention. B4.1: fixed action-type shortcut discovered. B4.2: action-type disambiguation. B5: epistemic-pragmatic closed-loop operation. B5.1: closed-loop degeneracy audit.

## 3. Audits

- per-episode plan overlap
- closed-loop policy provenance
- inspect/skip/intervene/skip diversity
- trace-update specificity
- feedback-revision specificity
- baseline sanity
- value leakage audit
- planning-budget stress

## 4. Results

| model | seed | cross_model_exact_plan_match_rate | exact_all_model_same_plan_rate | fixed_closed_loop_plan_rate | inspect_always_rate | intervene_immediately_rate | always_inspect_then_intervene_rate | skip_inspect_when_not_needed_rate | skip_intervention_when_not_needed_rate | decision_diversity_score | shared_closed_loop_policy_usage_rate | private_trace_closed_loop_usage_rate | fallback_usage_rate | oracle_plan_usage_rate | oracle_trace_update_usage_rate | oracle_feedback_revision_usage_rate | trace_update_specificity | trace_update_ablation_drop | trace_update_over_non_trace_ratio | wrong_inspection_update_drop | shuffled_inspection_update_drop | scripted_update_score | model_gain_over_scripted_update | feedback_revision_specificity | feedback_revision_ablation_drop | feedback_revision_over_scripted_ratio | contradictory_feedback_sensitivity | random_closed_loop_score | saliency_closed_loop_score | short_horizon_closed_loop_score | inspect_always_score | intervene_immediately_score | oracle_closed_loop_score | model_gain_over_inspect_always | model_gain_over_intervene_immediately | value_leakage_count | planning_budget_stress_retention | strict_budget_compliance | b51_closed_loop_audit_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 1.000 | 0.113 | 0.500 | 0.250 | 0.500 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 | 500000.000 | 0.500 | 0.656 | 1.000 | 0.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.006 | 0.125 | 0.125 | 0.225 | 0.500 | 1.000 | 0.775 | 0.500 | 800.000 | 1.000 | 1.000 | 0.000 |
| field_memory_model | 0 | 1.000 | 1.000 | 0.113 | 0.500 | 0.250 | 0.500 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 | 500000.000 | 0.500 | 0.656 | 1.000 | 0.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.006 | 0.125 | 0.125 | 0.225 | 0.500 | 1.000 | 0.775 | 0.500 | 800.000 | 1.000 | 1.000 | 0.000 |
| schema_memory_model | 0 | 1.000 | 1.000 | 0.113 | 0.500 | 0.250 | 0.500 | 1.000 | 1.000 | 0.750 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 | 500000.000 | 0.500 | 0.656 | 1.000 | 0.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.006 | 0.125 | 0.125 | 0.225 | 0.500 | 1.000 | 0.775 | 0.500 | 800.000 | 1.000 | 1.000 | 0.000 |

## 5. Interpretation

B5.1 shows that B5 should currently be interpreted as closed-loop path success under toy diagnostics, not genuine adaptive closed-loop operational structure.

## 6. Claim Boundary

Do not claim real control.
Do not claim robotics ability.
Do not claim engineering deployment.
Do not claim human-like active inference.
Do not claim language-free cognition solved.
