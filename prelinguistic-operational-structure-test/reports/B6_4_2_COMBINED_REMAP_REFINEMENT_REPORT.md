# B6.4.2 Combined Remap Refinement

## Purpose
B6.4.2 diagnoses the B6.4.1 combined_remap_hard oracle gap through pairwise, triple, and mechanism-ablation combined remaps.

## Starting Blocker
B6.4.1 combined_remap_hard had policy=0.675 and oracle_gap=0.325.

## Results
- combined_refinement_score = 0.805
- combined_policy_score = 0.675
- combined_oracle_gap = 0.325
- combined_failure_source = dynamics_delay_credit_interaction
- failure_attribution_confidence = 1.000

## Claim Boundary
B6.4.2 supports only toy-to-toy combined-remap diagnostic evidence. It does not support real-world transfer, robotics, construction-site autonomy, safety certification, or deployable control.
