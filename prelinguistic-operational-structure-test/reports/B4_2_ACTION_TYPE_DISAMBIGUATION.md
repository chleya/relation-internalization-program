# B4.2 Action-Type Disambiguation

## 1. Purpose

B4.1 found that B4 intervention success still used a fixed action type. B4.2 tests whether models can select different action types when action type matters.

## 2. Background

PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: active inspection degeneracy. B3.2: mechanism-disambiguating active inspection. B4: trace-guided intervention. B4.1: fixed action-type shortcut discovered. B4.2: action-type disambiguation.

## 3. Task Design

- action-type-specific intervention targets
- correct-region-wrong-action penalty
- family-action mapping stress
- action-type counterfactual
- fixed-action baseline
- action-type ablation
- action-type OOD

## 4. Models

- recurrent_flow_checkpoint_model
- field_memory_model
- schema_memory_model

## 5. Baselines

- fixed action
- random action type
- saliency
- short-horizon
- oracle action type

## 6. Results

| model | seed | fixed_action_type_rate | action_type_accuracy | region_accuracy | joint_region_action_accuracy | correct_region_wrong_action_penalty | action_type_counterfactual_sensitivity | family_action_diversity | family_action_mapping_accuracy | recurrent_action_accuracy | field_action_accuracy | schema_action_accuracy | action_type_shift_after_trace_ablation | action_type_ablation_drop | region_stability_after_action_ablation | gain_over_fixed_action_baseline | gain_over_random_action_type | gain_over_saliency | gain_over_short_horizon | action_type_ood_accuracy | oracle_action_type_score | value_leakage_count | b42_action_type_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 0.500 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.719 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.877 |
| field_memory_model | 0 | 0.500 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.819 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.877 |
| schema_memory_model | 0 | 0.500 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.738 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.937 |

## 7. Interpretation

B4.2 reduces the fixed-action-type shortcut discovered in B4.1. Under the current toy diagnostics, private delayed traces can guide not only intervention-region selection, but also differentiated action-type selection under action-type-specific intervention values.

## 8. Claim Boundary

Do not claim real control.
Do not claim robotics ability.
Do not claim engineering deployment.
Do not claim general active intelligence.
