# B6.4 Transfer and Anti-Overfit Generalization

## Purpose
B6.4 tests toy-to-toy transfer and anti-overfit generalization. It does not enter B7 or make real-world control claims.

## Remap Conditions
- visual_remap
- risk_cue_remap
- dynamics_remap
- delay_profile_remap
- indirect_path_remap
- mask_visibility_remap
- combined_remap

## Results
- b6_4_transfer_score = 0.752
- transfer_score = 1.000
- transfer_drop = 0.000
- baseline_transfer_gap = 0.193
- oracle_gap = 0.000
- combined_remap_score = 1.000

## Interpretation
B6.4 is a transfer diagnostic. Remap success supports only toy-to-toy operational structure transfer evidence. Remap failures indicate split-specific or shortcut-explainable behavior.

High transfer scores must be interpreted against baseline separation. If state_only, mask_only, or trace_only match the transfer policy on a remap, that split remains shortcut-explainable and should not be treated as strong operational transfer evidence.

## Claim Boundary
B6.4 supports only toy-to-toy transfer evidence. It does not support real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.
