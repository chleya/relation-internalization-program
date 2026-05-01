# B3.2 Mechanism-Disambiguating Active Inspection

## 1. Purpose

B3.1 showed that B3 active inspection is positive but not mechanism-disambiguating. B3.2 creates family-specific inspection targets to test whether recurrent / field / schema active inspection mechanisms can be separated.

## 2. Background

PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: active-inspection degeneracy. B3.2: mechanism-disambiguating active inspection.

## 3. Task Design

- family-specific inspection targets
- multi-objective inspection value
- non-linguistic goal code
- inspect value decomposition
- mechanism disagreement episodes
- family-specific trace ablation

## 4. Models

- recurrent_flow_checkpoint_model
- field_memory_model
- schema_memory_model

## 5. Baselines

- random
- saliency
- short-horizon
- oracle family inspection

## 6. Results

| model | seed | family_specific_inspection_accuracy | recurrent_goal_accuracy | field_goal_accuracy | schema_goal_accuracy | mechanism_disagreement_rate | cross_model_same_region_rate | task_conditioned_switch_accuracy | inspect_value_decomposition_alignment | recurrent_value_alignment | field_value_alignment | schema_value_alignment | family_specific_trace_ablation_drop | recurrent_trace_ablation_drop | field_trace_ablation_drop | schema_trace_ablation_drop | non_target_family_stability | gain_over_random | gain_over_saliency | gain_over_short_horizon | oracle_family_inspection_score | b32_mechanism_inspection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.977 | 0.633 | 0.633 | 1.000 | 0.963 |
| field_memory_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.977 | 0.633 | 0.633 | 1.000 | 0.963 |
| schema_memory_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.977 | 0.633 | 0.633 | 1.000 | 0.963 |

## 7. Interpretation

B3.2 reduces the B3.1 same-region degeneracy by constructing family-specific inspection targets and multi-objective inspection values. Under current toy diagnostics, recurrent / field / schema private traces show partial mechanism-disambiguated active inspection behavior.

## 8. Claim Boundary

Do not claim real active intelligence.
Do not claim real-world inspection ability.
Do not claim complete mechanism independence.
Do not claim language-free cognition solved.
