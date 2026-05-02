# B6.4 Adversarial Result Review

## Decision
- B6.4 first pass is a transfer harness pass, not strong transfer evidence.
- second_pass_needed: true

## Summary
- b64_policy_scores_all_remaps: True
- mean_baseline_transfer_gap: 0.193
- mean_oracle_gap: 0.000
- shortcut_equivalent_remap_count: 4
- shortcut_equivalent_remaps: ['dynamics_remap', 'mask_visibility_remap', 'risk_cue_remap', 'visual_remap']
- stronger_transfer_remaps: ['combined_remap', 'delay_profile_remap', 'indirect_path_remap']

## Remap-by-Remap Audit
- clean_reference: b64=1.000, state=1.000, mask=0.698, trace=1.000, oracle=1.000, shortcut_explainable=true, likely_shortcut_source=state_only, needed_hardening=reduce state_only shortcut
- combined_remap: b64=1.000, state=0.450, mask=0.450, trace=0.550, oracle=1.000, shortcut_explainable=false, likely_shortcut_source=none, needed_hardening=retain_as_transfer_diagnostic
- delay_profile_remap: b64=1.000, state=0.550, mask=0.550, trace=0.550, oracle=1.000, shortcut_explainable=false, likely_shortcut_source=none, needed_hardening=retain_as_transfer_diagnostic
- dynamics_remap: b64=1.000, state=0.450, mask=0.698, trace=1.000, oracle=1.000, shortcut_explainable=true, likely_shortcut_source=trace_only, needed_hardening=alter dynamics enough that trace-only and conservative rules fail
- indirect_path_remap: b64=1.000, state=0.550, mask=0.450, trace=0.550, oracle=1.000, shortcut_explainable=false, likely_shortcut_source=none, needed_hardening=retain_as_transfer_diagnostic
- mask_visibility_remap: b64=1.000, state=1.000, mask=0.450, trace=1.000, oracle=1.000, shortcut_explainable=true, likely_shortcut_source=state_only, needed_hardening=hide answer-like mask fields plus state/trace substitutes
- risk_cue_remap: b64=1.000, state=1.000, mask=0.698, trace=1.000, oracle=1.000, shortcut_explainable=true, likely_shortcut_source=state_only, needed_hardening=hide direct risk estimate and require feedback/history risk inference
- visual_remap: b64=1.000, state=0.450, mask=0.698, trace=1.000, oracle=1.000, shortcut_explainable=true, likely_shortcut_source=trace_only, needed_hardening=remove trace-equivalent visual/state target cue and add appearance distractors

## Shortcut Audit
- visual_shortcut_source: trace_only
- risk_cue_shortcut_source: state_only
- dynamics_shortcut_source: trace_only
- mask_visibility_shortcut_source: state_only
- shortcut_equivalent_remap_count: 4

## Stronger Remap Audit
- delay_profile_remap: {'present': True, 'baseline_transfer_gap': 0.44999999999999996, 'oracle_gap': 0.0, 'mechanism_readout': 'delayed_credit_or_hidden_indirect_history', 'interpretation': 'stronger harness signal, still synthetic toy transfer'}
- indirect_path_remap: {'present': True, 'baseline_transfer_gap': 0.44999999999999996, 'oracle_gap': 0.0, 'mechanism_readout': 'delayed_credit_or_hidden_indirect_history', 'interpretation': 'stronger harness signal, still synthetic toy transfer'}
- combined_remap: {'present': True, 'baseline_transfer_gap': 0.44999999999999996, 'oracle_gap': 0.0, 'mechanism_readout': 'delayed_credit_or_hidden_indirect_history', 'interpretation': 'stronger harness signal, still synthetic toy transfer'}

## Integrity
- forbidden_reference_count_max: 0
- poisoned_evaluator_invariance_all_pass: True
- no_sample_metric_count_total: 0
- invalid_metric_count_total: 0
- remap_leakage_count_total: 0
- selected_source_counts: {'feedback': 30, 'delayed_credit': 12, 'hidden_indirect_history': 6}
- review_claim_boundary: toy_to_toy_transfer_only

## Caveats
- B6.4 scores 1.000 on all remaps, which is too clean for strong transfer evidence.
- mean_baseline_transfer_gap is modest, so shortcut baselines still explain part of the result.
- visual_remap, risk_cue_remap, dynamics_remap, and mask_visibility_remap remain shortcut-explainable.
- delay_profile_remap, indirect_path_remap, and combined_remap show stronger baseline separation, but remain synthetic toy diagnostics.
- No real-world generalization, robotics, safety certification, construction-site autonomy, or deployment claim is supported.
