# B6.1 Audit Notes

## Purpose

This audit checks whether B6.1 high scores and degradation points are caused by metric defaults, evaluator leakage, clean mask dependence, or superficial scoring artifacts.

## Metric Integrity

- B6.1 metric means return 0.0 on empty records.
- Wrong decisions can score below 1.0 through safety and utility penalties.
- The known B6 legacy inspectable metric shortcut is not used by B6.1 scoring.

## Policy Input Boundary

- hardening_policy forbidden reference count: 0
- policy uses model_input only: True
- Indirect target remains public in model_input; delayed-indirect stress should be interpreted as delayed/backfire handling, not indirect-path discovery.

## Missing Mask = 1.0

- risk_constrained_score: 0.833
- safety_score: 1.000
- utility_score: 0.667
- abstain_rate: 0.500
- unsafe_action_rate: 0.000
- false_safe_commit_rate: 0.000
- gain_over_mask_only: 0.250
- gain_over_risk_blind: 0.208
- gap_to_oracle: 0.167

Interpretation: missing mask causes a real score and utility drop. The fallback still beats risk-blind/mask-only in this run, but the small margin over mask-only means this is not strong evidence of private trace risk inference.

## Delayed Indirect delay_steps = 5

- risk_constrained_score: 0.725
- delayed_indirect_success_rate: 0.000
- delayed_indirect_credit_assignment_accuracy: 0.000
- premature_direct_action_rate: 0.000
- backfire_avoidance_accuracy: 0.000
- gap_to_oracle: 0.275

Interpretation: delay_steps=5 exposes a real delayed/backfire weakness. Because the indirect target is still public, this is not evidence of discovering hidden indirect channels.

## Baseline Strength

- mean_gain_over_risk_blind: 0.338
- mean_gain_over_mask_only: 0.175
- mean_gap_to_oracle: 0.045
- random_mean: 0.724
- mask_only_close_to_hardening: False

Random baseline remains non-trivial, suggesting that some B6.1 stress conditions are partially solvable through conservative or chance-level behavior. Therefore B6.1 should be interpreted as a diagnostic benchmark rather than strong proof of robust risk intelligence.

## Delayed Indirect Caveat

The aggregate delayed-indirect score remains non-zero because the policy avoids unsafe direct actions, but the delayed-indirect subtask itself fails under delay_steps=5. This is a genuine credit-assignment weakness, not a solved delayed intervention result.

## Conservative Conclusion

B6.1 is a useful reviewer-hardening stress diagnostic, but it still relies on public operational cues including mask fields, latent risk markers, risk-history signals, and public indirect target candidates. It does not establish real-world safety intelligence.
