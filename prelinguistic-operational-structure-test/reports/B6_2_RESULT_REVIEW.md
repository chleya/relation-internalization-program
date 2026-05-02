# B6.2 Result Review

## Purpose

This review audits B6.2 second-pass outputs for split-level failures, baseline explanations, hidden leakage risks, and metric integrity. It does not add new capability and does not replace B6/B6.1 results.

## Source Summary

- conditions: ambiguous_trace, delayed_indirect, hard_hidden_mask, hidden_irreversibility, hide_indirect_target, low_confidence_trace, missing_mask, missing_trace, risk_reward_conflict, spurious_flip, test, wrong_trace, wrong_trace_state_ambiguous
- policies: always_abstain, b62_policy, conservative_uncertainty, mask_only, oracle, random, risk_blind, state_only, trace_only
- missing requested focus conditions: none
- b62_policy_mean_score: 0.976
- oracle_mean_score: 1.000
- random_mean_score: 0.561

## Split-Level Audit

| condition | b62 score | utility | safety | gap to oracle | gain mask_only | gain state_only | gain trace_only | samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ambiguous_trace | 1.000 | 1.000 | 1.000 | 0.000 | 0.247 | 0.247 | 0.373 | 12 |
| delayed_indirect | 1.000 | 1.000 | 1.000 | 0.000 | 0.433 | 0.504 | 1.000 | 12 |
| hard_hidden_mask | 1.000 | 1.000 | 1.000 | 0.000 | 0.583 | 0.321 | 0.583 | 12 |
| hidden_irreversibility | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 12 |
| hide_indirect_target | 1.000 | 1.000 | 1.000 | 0.000 | 0.583 | 0.321 | 0.583 | 12 |
| low_confidence_trace | 1.000 | 1.000 | 1.000 | 0.000 | 0.247 | 0.247 | 0.373 | 12 |
| missing_mask | 1.000 | 1.000 | 1.000 | 0.000 | 0.550 | 0.396 | 0.583 | 12 |
| missing_trace | 1.000 | 1.000 | 1.000 | 0.000 | 0.247 | 0.247 | 0.458 | 12 |
| risk_reward_conflict | 1.000 | 1.000 | 1.000 | 0.000 | 0.247 | 0.550 | 1.000 | 12 |
| spurious_flip | 1.000 | 1.000 | 1.000 | 0.000 | 0.583 | 0.321 | 0.583 | 12 |
| test | 0.938 | 0.888 | 1.000 | 0.062 | 0.000 | 0.000 | 0.000 | 12 |
| wrong_trace | 0.753 | 0.550 | 1.000 | 0.247 | -0.081 | -0.247 | 0.035 | 12 |
| wrong_trace_state_ambiguous | 1.000 | 1.000 | 1.000 | 0.000 | 0.310 | 0.247 | 0.247 | 12 |

## wrong_trace Audit

- b62_policy score: 0.753
- mask_only score: 0.833
- state_only score: 1.000
- trace_only score: 0.717
- conservative score: 0.592
- oracle score: 1.000
- gain_over_mask_only: -0.081
- gain_over_state_only: -0.247
- trace_region_reliance_score: 0.247
- wrong_trace_failure_rate: 0.000
- trace_confidence_calibration: 1.000
- fallback_under_trace_uncertainty_score: 0.753
- abstain_rate: 0.167
- unsafe_action_rate: 0.000
- false_safe_commit_rate: 0.000

Conclusion: wrong_trace remains an unresolved structural weakness. B6.2 does not yet prove autonomous target correction or robust trace repair, because b62_policy does not outperform mask_only or state_only on this split.

## wrong_trace_state_ambiguous Audit

- b62_policy score: 1.000
- state_only score: 0.753
- mask_only score: 0.690
- state_only_drop_on_ambiguous_trace: 0.247
- inspect_recovery_rate: 1.000
- trace_repair_under_ambiguous_state_score: 1.000

Conclusion: wrong_trace_state_ambiguous checks whether state_only stops being a perfect explanation when the public state cue is noisy.

## hide_indirect_target Audit

- candidate_indirect_search_success_rate: 0.583
- delayed_indirect_success_rate: 0.000
- delayed_indirect_credit_assignment_accuracy: 0.000
- backfire_avoidance_accuracy: 0.000
- indirect_target_dependency_score: 0.000
- public_mask_dependency_score: 0.000
- hidden_mask_performance_drop: 0.000
- gap_to_oracle: 0.000
- failure_reason distribution: {'none': 12}

Conclusion: hide_indirect_target only tests candidate-search fallback. It does not prove hidden indirect causal path discovery unless candidate search succeeds without public indirect cues and without oracle target leakage.

## missing_mask Audit

- B6.1 baseline utility_score reference: 0.667
- B6.2 utility_score: 1.000
- safety_score: 1.000
- abstain_rate: 0.000
- unnecessary_abstain_rate: 0.000
- unsafe_action_rate: 0.000
- false_safe_commit_rate: 0.000
- missing_mask_utility_recovery: 0.333
- missing_mask_gap_to_oracle: 0.000
- gain_over_mask_only: 0.550
- gain_over_state_only: 0.396
- gain_over_conservative: 0.321

Conclusion: missing_mask improves relative to the B6.1 reference and stays safe, but this remains a state-estimate fallback diagnostic rather than proof that private trace alone infers risk.

## delayed_indirect Audit

| delay_steps | score | delayed success | credit accuracy | backfire avoidance | premature direct | unnecessary wait | cap count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0 |
| 2 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0 |
| 3 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0 |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0 |

Conclusion: delay_steps=5 now includes actual delayed indirect successes, but the split also contains backfire-avoidance cases. The aggregate score must be interpreted together with delay5_true_success_score rather than as solved delayed intervention.

## Baseline Sanity

| policy | mean score | mean utility | mean safety | gap to oracle |
| --- | ---: | ---: | ---: | ---: |
| b62_policy | 0.976 | 0.957 | 1.000 | 0.024 |
| risk_blind | 0.462 | 0.405 | 0.532 | 0.538 |
| mask_only | 0.595 | 0.526 | 0.769 | 0.405 |
| random | 0.561 | 0.424 | 0.782 | 0.439 |
| always_abstain | 0.518 | 0.295 | 1.000 | 0.482 |
| oracle | 1.000 | 1.000 | 1.000 | 0.000 |
| trace_only | 0.452 | 0.381 | 0.558 | 0.548 |
| state_only | 0.734 | 0.589 | 0.987 | 0.266 |
| conservative_uncertainty | 0.642 | 0.463 | 1.000 | 0.358 |

- random_mean_remains_nontrivial: True
- mask_only_remains_strong: False
- state_only_explains_wrong_trace: True

## Metric Integrity

- no_sample_metric_count_total: 0
- invalid_metric_count_total: 0
- forbidden_reference_count_max: 0
- poisoned_ground_truth_invariance_all_pass: True
- policy_uses_model_input_only_all_pass: True
- empty_focus_conditions: []

No new metric bug was found in this review. The existing B6.2 regression tests prevent wrong decisions and empty metrics from silently receiving full credit. Any missing standalone focus split is listed above and must not be reported as passed.

## Overall Conclusion

- No new evaluator-ground-truth leakage or constant-one metric bug was found in this result review.
- wrong_trace remains the central unresolved weakness: b62_policy does not beat mask_only/state_only, so B6.2 does not prove robust trace repair.
- hide_indirect_target succeeds as candidate search fallback, not as proof of hidden indirect causal path discovery.
- missing_mask utility improves by 0.333 over the B6.1 reference while preserving safety.
- delay_steps=5 now includes actual delayed indirect successes, but it must still be interpreted together with backfire-avoidance cases.
- random remains non-trivial, so aggregate scores still include partial credit available to weak or chance policies.
- Trace correction remains the next bottleneck before stronger claims.
- wrong_trace_state_ambiguous reduces the state_only shortcut compared with the first-pass wrong_trace split.

## Claim Boundary

B6.2 second pass supports only a toy diagnostic claim: fallback-risk and delayed-credit stress tests can be run and can expose weaknesses. It does not prove robust fallback risk inference, autonomous trace repair, hidden indirect causal path discovery, real-world risk intelligence, safety certification, robotics capability, or deployable control.
