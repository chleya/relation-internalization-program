# B4.1 Intervention Degeneracy Audit

## 1. Purpose

B4 produced strong intervention/action-selection results. B4.1 audits whether this reflects private trace-guided intervention or shared action policy / shortcuts / leakage / weak baselines.

## 2. Background

PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: active inspection degeneracy audit. B3.2: mechanism-disambiguating active inspection. B4: delayed trace-guided intervention. B4.1: intervention degeneracy audit.

## 3. Audits

- per-episode action overlap
- action policy provenance
- family-specific action scorer
- shared action policy ablation
- wrong-action / wrong-region stress
- baseline sanity
- action-after-trace-ablation specificity
- intervention value leakage check

## 4. Results

| model | seed | cross_model_exact_action_match_rate | exact_all_model_same_action_rate | fixed_action_type_rate | action_type_entropy | shared_action_policy_usage_rate | private_trace_action_score_usage_rate | fallback_usage_rate | oracle_value_usage_rate | action_scorer_specificity | mean_action_scorer_correlation | shared_action_policy_ablation_drop | private_action_retention_after_shared_ablation | wrong_action_penalty | wrong_region_penalty | wrong_action_wrong_region_penalty | random_intervention_score | saliency_intervention_score | short_horizon_intervention_score | inspect_only_score | oracle_intervention_score | trace_over_saliency_gain_margin | trace_over_short_horizon_gain_margin | trace_over_inspect_only_gain_margin | private_trace_ablation_drop | matched_non_trace_ablation_drop | private_trace_over_non_trace_ratio | action_type_shift_after_trace_ablation | region_shift_after_trace_ablation | non_trace_action_stability | value_leakage_count | b41_intervention_audit_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.945 | 0.055 | 0.000 | 1.000 | 0.650 | 0.900 | 1.000 | 0.006 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1000000.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| field_memory_model | 0 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.945 | 0.055 | 0.000 | 1.000 | 0.650 | 0.900 | 1.000 | 0.006 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1000000.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| schema_memory_model | 0 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.945 | 0.055 | 0.000 | 1.000 | 0.650 | 0.900 | 1.000 | 0.006 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1000000.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |

## 5. Interpretation

B4.1 shows that B4 should currently be interpreted as trace-guided intervention-path success, not independently validated recurrent / field / schema intervention mechanisms. The failure localizes the next bottleneck to action-policy degeneracy, action-value leakage, weak action baselines, or nonspecific trace ablation.

## 6. Claim Boundary

Do not claim real control.
Do not claim robotics ability.
Do not claim engineering deployment.
Do not claim general active intelligence.
Do not claim language-free cognition solved.
