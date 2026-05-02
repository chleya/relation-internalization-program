# B6.4.2 Result Review

## Decision
- submit_ready_as_diagnostic: true
- gate_judgment: STAY_IN_B6_REFINEMENT
- gate_reason: combined_remap_hard remains a B6.x refinement blocker

## Pairwise Results
- pair_visual_risk: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present
- pair_visual_dynamics: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present
- pair_risk_mask: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present
- pair_dynamics_delay: policy=0.887, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.537, oracle_gap=0.113, reason=baseline_separation_present
- pair_mask_indirect: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present
- pair_delay_indirect: policy=0.887, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.537, oracle_gap=0.113, reason=baseline_separation_present

## Triple Results
- triple_visual_risk_dynamics: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present
- triple_risk_mask_delay: policy=0.887, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.537, oracle_gap=0.113, reason=baseline_separation_present
- triple_dynamics_delay_indirect: policy=0.675, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.325, oracle_gap=0.325, reason=large_oracle_gap
- triple_visual_mask_indirect: policy=0.838, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.488, oracle_gap=0.162, reason=baseline_separation_present
- triple_visual_risk_mask: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present

## Mechanism Ablations
- combined_disable_trace_repair: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present
- combined_disable_fallback_risk: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present
- combined_disable_delayed_credit: policy=0.350, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.000, oracle_gap=0.650, reason=mechanism_removed_policy_collapses
- combined_disable_candidate_search: policy=0.350, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.000, oracle_gap=0.650, reason=mechanism_removed_policy_collapses
- combined_disable_inspection_recovery: policy=1.000, state=0.350, mask=0.350, trace=0.250, oracle=1.000, gap=0.650, oracle_gap=0.000, reason=baseline_separation_present

## Failure Attribution
- combined_failure_source: dynamics_delay_credit_interaction
- combined_oracle_gap: 0.325

## Integrity
- forbidden_reference_count_max: 0
- invalid_metric_count_total: 0
- no_sample_metric_count_total: 0
- combined_leakage_count_total: 0
- shortcut_leakage_count_total: 0
- poisoned_evaluator_invariance_all_pass: True

## Remaining Blockers
- combined_remap_hard oracle gap remains 0.325

## Claim Boundary
B6.4.2 is toy-to-toy combined-remap diagnostic evidence only. It does not support real-world transfer, robotics, safety certification, construction-site autonomy, or deployable control.
