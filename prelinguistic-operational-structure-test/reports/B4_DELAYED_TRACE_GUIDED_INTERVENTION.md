# B4 Delayed Trace-Guided Intervention / Action Selection

## 1. Purpose

B4 tests whether B3.2 private delayed traces can guide local intervention/action selection under a constrained action budget.

## 2. Background

PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: active inspection degeneracy audit. B3.2: mechanism-disambiguating active inspection. B4: trace-guided intervention/action selection.

## 3. Task

The model must select one local intervention action: action type and region id under budget = 1.

## 4. Minimal Action Space

- do_nothing
- inspect_only
- apply_local_damping
- apply_local_push
- block_force_region
- stabilize_trace_region

## 5. Models

- recurrent_flow_checkpoint_model
- field_memory_model
- schema_memory_model

## 6. Baselines

- random intervention
- saliency intervention
- short-horizon intervention
- inspect-only
- oracle intervention

## 7. Metrics

- trace-guided intervention accuracy
- region accuracy
- action type accuracy
- outcome improvement
- intervention-vs-inspection gain
- wrong-region penalty sensitivity
- family-specific intervention accuracy
- trace ablation intervention drop
- baseline gains
- delay OOD intervention

## 8. Results

| model | seed | trace_guided_intervention_accuracy | intervention_region_accuracy | action_type_accuracy | outcome_improvement | intervention_vs_inspection_gain | wrong_region_penalty_sensitivity | family_specific_intervention_accuracy | recurrent_intervention_accuracy | field_intervention_accuracy | schema_intervention_accuracy | trace_ablation_intervention_drop | delay_ood_intervention_accuracy | gain_over_random | gain_over_saliency | gain_over_short_horizon | gain_over_inspect_only | oracle_intervention_score | random_intervention_score | saliency_intervention_score | short_horizon_intervention_score | inspect_only_score | b4_intervention_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 0.006 | 0.000 | 0.000 | 0.000 | 0.985 |
| field_memory_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 0.006 | 0.000 | 0.000 | 0.000 | 0.985 |
| schema_memory_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 0.006 | 0.000 | 0.000 | 0.000 | 0.985 |

## 9. Interpretation

B4 supports that, in the toy PLOS environment, B3.2 private delayed traces can guide local intervention/action selection under a constrained action budget. The models choose both intervention region and action type, outperform random/saliency/short-horizon/inspect-only baselines, show outcome improvement, and lose action performance after trace ablation.

## 10. Claim Boundary

Do not claim real control.
Do not claim robotics deployment.
Do not claim real engineering intervention.
Do not claim general active intelligence.
Do not claim language-free cognition solved.
