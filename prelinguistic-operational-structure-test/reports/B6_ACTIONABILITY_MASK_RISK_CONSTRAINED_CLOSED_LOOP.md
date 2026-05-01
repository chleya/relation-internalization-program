# B6 Actionability Mask / Risk-Constrained Closed Loop

## 1. Purpose

B6 tests whether the B5.2 clean closed-loop system can operate under actionability, risk, irreversibility, and cost constraints.

## 2. Background

PLOS v1 through B4.2 established delayed trace to inspect/intervene/action type. B5-Clean fixed oracle/value leakage. B5.2 showed content-sensitive update and feedback revision. B6 adds actionability mask and risk-constrained choice.

## 3. Actionability Mask

- observable
- inspectable
- directly_intervenable
- indirectly_intervenable
- unsafe
- irreversible
- costly

## 4. Results

| model | seed | value_leakage_count | oracle_actionability_usage_rate | oracle_risk_value_usage_rate | actionability_mask_accuracy | inspectable_decision_accuracy | direct_intervention_accuracy | indirect_intervention_accuracy | unsafe_action_rejection_rate | irreversible_action_rejection_rate | costly_action_avoidance_accuracy | abstain_when_required_accuracy | act_when_safe_and_needed_accuracy | wrong_actionability_penalty_sensitivity | risk_adjusted_value_alignment | cost_sensitive_planning_accuracy | risk_aware_feedback_revision_accuracy | unsafe_feedback_correction_rate | irreversible_feedback_correction_rate | gain_over_random | gain_over_saliency | gain_over_short_horizon | gain_over_risk_blind | gain_over_always_act | gain_over_always_abstain | oracle_risk_constrained_score | random_risk_constrained_score | b6_risk_constrained_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 0.857 | 0.857 | 0.714 | 1.000 | 0.100 | 0.986 |
| field_memory_model | 0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 0.857 | 0.857 | 0.714 | 1.000 | 0.100 | 0.986 |
| schema_memory_model | 0 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 0.857 | 0.857 | 0.714 | 1.000 | 0.100 | 0.986 |

## 5. Interpretation

B6 supports that, in the toy PLOS environment, the clean B5.2 closed-loop system can use an actionability mask to choose inspect, direct intervention, indirect intervention, or abstain under risk and cost constraints.

## 6. Claim Boundary

Do not claim real control.
Do not claim robotics ability.
Do not claim engineering deployment.
Do not claim safety-certified planning.
Do not claim human-like risk reasoning.
