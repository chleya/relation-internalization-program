# B6.4.1 Adversarial Result Review

## Decision
- B6.4.1 is submit-ready as a diagnostic branch.
- It is not a general transfer proof.

## Summary
- hard_transfer_score: 0.935
- hard_baseline_transfer_gap: 0.605
- hard_oracle_gap: 0.065
- shortcut_equivalent_hard_remap_count: 0
- transfer_evidence_strength: 0.727

## Hard Remap Audit
- visual_remap_hard: policy=1.000, state=0.350, mask=0.250, trace=0.250, random=0.283, always_abstain=0.350, conservative=0.350, oracle=1.000, oracle_gap=0.000, baseline_gap=0.650, hardening_success=true, failure_reason=none
- risk_cue_remap_hard: policy=1.000, state=0.250, mask=0.250, trace=0.250, random=0.283, always_abstain=0.350, conservative=0.350, oracle=1.000, oracle_gap=0.000, baseline_gap=0.750, hardening_success=true, failure_reason=none
- dynamics_remap_hard: policy=1.000, state=0.350, mask=0.250, trace=0.250, random=0.283, always_abstain=0.350, conservative=0.350, oracle=1.000, oracle_gap=0.000, baseline_gap=0.650, hardening_success=true, failure_reason=none
- mask_visibility_remap_hard: policy=1.000, state=0.350, mask=0.350, trace=0.250, random=0.283, always_abstain=0.350, conservative=0.350, oracle=1.000, oracle_gap=0.000, baseline_gap=0.650, hardening_success=true, failure_reason=none
- combined_remap_hard: policy=0.675, state=0.350, mask=0.350, trace=0.250, random=0.283, always_abstain=0.350, conservative=0.350, oracle=1.000, oracle_gap=0.325, baseline_gap=0.325, hardening_success=true, failure_reason=limited_transfer_oracle_gap

## Hidden Cue / Leakage Audit
- hidden_answer_cue_count: 0
- public_state_shortcut_leakage_count: 0
- mask_answer_leakage_count: 0
- indirect_target_leakage_count: 0
- dynamics_shortcut_leakage_count: 0
- combined_failure_reason_distribution: {'delayed_credit': 3, 'abstain_uncertain': 3}
- b641_transfer_source_counts: {'feedback': 24, 'hidden_indirect': 6, 'delayed_credit': 3, 'abstain_uncertain': 3}

## Baseline Drop Audit
- state_only_drop_by_remap: {'visual_remap_hard': 0.6500000000000001, 'risk_cue_remap_hard': 0.75, 'dynamics_remap_hard': 0.6500000000000001, 'mask_visibility_remap_hard': 0.6500000000000001, 'combined_remap_hard': 0.6500000000000001}
- mask_only_drop_by_remap: {'visual_remap_hard': 0.75, 'risk_cue_remap_hard': 0.75, 'dynamics_remap_hard': 0.75, 'mask_visibility_remap_hard': 0.6500000000000001, 'combined_remap_hard': 0.6500000000000001}
- trace_only_drop_by_remap: {'visual_remap_hard': 0.75, 'risk_cue_remap_hard': 0.75, 'dynamics_remap_hard': 0.75, 'mask_visibility_remap_hard': 0.75, 'combined_remap_hard': 0.75}
- random_score_by_remap: {'visual_remap_hard': 0.2833333333333333, 'risk_cue_remap_hard': 0.2833333333333333, 'dynamics_remap_hard': 0.2833333333333333, 'mask_visibility_remap_hard': 0.2833333333333333, 'combined_remap_hard': 0.2833333333333333}
- always_abstain_score_by_remap: {'visual_remap_hard': 0.3499999999999999, 'risk_cue_remap_hard': 0.3499999999999999, 'dynamics_remap_hard': 0.3499999999999999, 'mask_visibility_remap_hard': 0.3499999999999999, 'combined_remap_hard': 0.3499999999999999}
- oracle_score_by_remap: {'visual_remap_hard': 1.0, 'risk_cue_remap_hard': 1.0, 'dynamics_remap_hard': 1.0, 'mask_visibility_remap_hard': 1.0, 'combined_remap_hard': 1.0}
- baseline_drops_valid: True

## Metric Integrity
- forbidden_reference_count_max: 0
- invalid_metric_count_total: 0
- no_sample_metric_count_total: 0
- remap_leakage_count_total: 0
- shortcut_leakage_count_total: 0
- poisoned_evaluator_invariance_all_pass: True

## Combined Remap Conclusion
combined_remap_hard shows limited transfer with meaningful oracle gap; preserve as unresolved hard-transfer limit.

## Remaining Blockers
- combined_remap_hard remains limited by oracle gap.

## Claim Boundary
B6.4.1 is stronger than B6.4 first pass because shortcut-equivalent hard remap count dropped to 0. It remains toy-to-toy hard-remap diagnostic evidence only.

It does not support real-world transfer, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.
