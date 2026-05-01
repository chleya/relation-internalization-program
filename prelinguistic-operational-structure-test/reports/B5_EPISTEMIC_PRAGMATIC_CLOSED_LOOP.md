# B5 Epistemic-Pragmatic Closed-Loop Operation

## 1. Purpose

B5 tests whether B4.2 private delayed traces can support a two-step closed loop: observe -> inspect -> update trace -> intervene -> observe consequence -> revise trace.

## 2. Background

PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: inspection degeneracy audit. B3.2: mechanism-disambiguating active inspection. B4: trace-guided intervention. B4.1: fixed action-type shortcut discovered. B4.2: action-type disambiguation. B5: epistemic-pragmatic closed-loop operation.

## 3. Closed-loop Task

- observe
- inspect or skip
- update trace
- intervene or skip
- observe consequence
- revise trace

## 4. Epistemic vs Pragmatic Value

- epistemic value = information gain from inspection
- pragmatic value = outcome improvement from intervention

## 5. Baselines

- random
- saliency
- short-horizon
- inspect-always
- intervene-immediately
- oracle

## 6. Results

| model | seed | inspect_timing_accuracy | epistemic_value_alignment | trace_update_accuracy | trace_uncertainty_reduction | post_inspection_intervention_accuracy | pragmatic_value_alignment | feedback_revision_accuracy | closed_loop_gain_over_inspect_always | closed_loop_gain_over_intervene_immediately | closed_loop_gain_over_random | closed_loop_gain_over_saliency | closed_loop_gain_over_short_horizon | wrong_inspect_penalty_sensitivity | wrong_intervention_penalty_sensitivity | planning_budget_compliance | oracle_closed_loop_score | random_closed_loop_score | b5_closed_loop_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 1.000 | 0.775 | 0.500 | 0.982 | 0.875 | 0.875 | 1.000 | 0.750 | 1.000 | 1.000 | 0.018 | 0.887 |
| field_memory_model | 0 | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 1.000 | 0.775 | 0.500 | 0.989 | 0.875 | 0.875 | 1.000 | 0.750 | 1.000 | 1.000 | 0.011 | 0.887 |
| schema_memory_model | 0 | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 1.000 | 0.775 | 0.500 | 0.994 | 0.875 | 0.875 | 1.000 | 0.750 | 1.000 | 1.000 | 0.006 | 0.887 |

## 7. Interpretation

B5 supports that, in the toy PLOS environment, private delayed operational trace can support a minimal epistemic-pragmatic closed loop: the system can decide when to inspect, update trace from inspection, intervene based on updated trace, observe consequence, and revise trace under budget.

## 8. Claim Boundary

Do not claim real control.
Do not claim robotics capability.
Do not claim engineering deployment.
Do not claim human-like active inference.
Do not claim language-free cognition solved.
