# B6.3.1 Result Review

## Decision
- submit_ready_as_diagnostic: true

## Wrong Trace Audit
- wrong_trace_no_public_state_score: 1.000
- state_only_drop_on_wrong_trace_no_public_state: 0.550
- trace_conflict_resolution_accuracy: 1.000
- trace_confidence_downgrade_rate: 0.000
- repair_source_distribution: history:6
- inspect_recovery_rate: 0.275
- feedback_repair_gain: 0.275
- history_repair_gain: 0.275
- unsafe_action_after_wrong_trace_rate: 0.000
- conclusion: diagnostic_pressure_improved_but_not_general_robust_trace_repair

## Feedback / History Necessity Audit
- feedback_required_score: 1.000
- freeze_feedback_drop_on_feedback_required: 0.275
- history_required_score: 1.000
- remove_history_drop_on_history_required: 0.550
- feedback_update_necessity_evidence: 0.275
- history_necessity_evidence: 0.550
- substitute_cue_after_history_removed: mask_only remains strong in history_required_indirect_discovery; interpret necessity as split-specific.
- substitute_cue_after_feedback_frozen: history remains available in some feedback conditions; interpret feedback necessity narrowly.

## Delayed Credit Buffer Audit
- credit_buffer_required_score: 1.000
- drop_under_disable_credit_buffer_on_delay5_required: 0.550
- multiple_pending_credit_assignment_accuracy: 1.000
- delayed_success_vs_no_effect_accuracy: 1.000
- staged_backfire_credit_accuracy: 0.725
- delay5_credit_buffer_necessity_evidence: 0.550
- abstain_counted_as_success_rate: 0.000
- no_effect_counted_as_success_rate: 0.000
- backfire_counted_as_success_rate: 0.000
- conclusion: credit_buffer_necessity_strengthened_in_synthetic_delay5_required_splits

## Hidden Indirect Discovery Audit
- hidden_indirect_discovery_score: 1.000
- candidate_search_fallback_score: 0.000
- exploration_success_rate: 1.000
- spurious_candidate_rejection_rate: 1.000
- history_based_indirect_discovery_accuracy: 1.000
- hidden_indirect_gap_to_oracle: 0.000
- oracle_target_leakage_count: 0
- public_indirect_target_leakage_count: 0
- candidate_shortcut_leakage_count: 0
- exploration_required_sample_count: 1
- spurious_candidate_sample_count: 1
- synthetic_outcome_history_target_cue_count: 4
- conclusion: synthetic_exploration_history_diagnostic_not_real_world_causal_discovery

## Baseline Sanity
- always_abstain: 0.502
- b63_1_no_candidate_search: 0.856
- b63_1_no_credit_buffer: 0.895
- b63_1_no_feedback_update: 0.935
- b63_1_no_history: 0.750
- b63_1_no_inspection_recovery: 0.974
- b63_1_policy: 0.987
- conservative_uncertainty: 0.489
- mask_only: 0.654
- oracle: 1.000
- random: 0.607
- state_only: 0.515
- trace_only: 0.725

## Leakage / Metric Integrity
- forbidden_reference_count_max: 0
- invalid_metric_count_total: 0
- no_sample_metric_count_total: 0
- poisoned_ground_truth_invariance_all_pass: True
- policy_source_forbidden_reference_count: 0
- missing_mandatory_conditions: []
- zero_sample_conditions: []
- abstain_counted_as_success_rate: 0.000
- no_effect_counted_as_success_rate: 0.000
- backfire_counted_as_success_rate: 0.000

## Remaining Blockers
- No blocking diagnostic issue found for submit-ready diagnostic branch.

## Caveats
- B6.3.1 is a diagnostic branch, not a solved robustness claim.
- wrong_trace pressure improved, but robust trace repair is not generally solved.
- feedback/history necessity is supported only in synthetic required splits.
- delayed credit buffer necessity is strengthened only in delay5-required diagnostics.
- hidden indirect discovery uses synthetic exploration/outcome-history cues; it is not real-world causal discovery.
- high aggregate score should not be treated as general structural necessity proof.
- No real-world risk intelligence, robotics, safety certification, construction-site autonomy, or engineering deployment claim is supported.
