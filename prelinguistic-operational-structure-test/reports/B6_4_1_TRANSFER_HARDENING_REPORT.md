# B6.4.1 Transfer Hardening

## Purpose
B6.4.1 hardens B6.4 first-pass remaps that were shortcut-explainable.

## Results
- hard_transfer_score = 0.935
- hard_baseline_transfer_gap = 0.605
- shortcut_equivalent_hard_remap_count = 0
- transfer_evidence_strength = 0.727

## Claim Boundary
B6.4.1 supports only toy-to-toy hard-remap diagnostic evidence. It does not support real-world generalization, robotics, construction-site autonomy, safety certification, or deployable control.

combined_remap_hard remains limited when its oracle gap is non-trivial. A lower combined score should be treated as a hard-transfer limitation, not hidden by the aggregate score.
