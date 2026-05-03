# G1 Minimal Generator Adversarial Review

## Decision
- decision: submit_ready_as_minimal_generator_branch_with_caveats
- leakage_issue_found: false
- metric_issue_found: false

## Score Audit
- g1_ood_score = 0.868
- g1_ood_gain_over_random = 0.637
- g1_ood_gain_over_hand_designed = 0.273
- g1_oracle_gap = 0.110
- g1_mask_f1 = 0.934

## Pressure Usage Audit
- uses_prediction_error: true
- uses_intervention_gain: true
- uses_risk_proxy: true
- uses_feedback: false
- uses_compression: false

## Interpretation
G1 is a valid minimal generator start because it selects a compact rule from interaction history and beats weak baselines on held-out OOD remap.
However, the selected rule does not use feedback or compression pressure. This means G1 has not yet shown that all intended pressures are necessary for generated operational structure.

## Remaining Blockers
- selected rule does not use all intended pressure channels: ['uses_feedback', 'uses_compression']
- feature vocabulary and rule family remain hand-scaffolded
- OOD remap is synthetic and does not prove autonomous structure discovery

## Claim Boundary
G1 supports only a minimal toy generator diagnostic. It does not prove autonomous cognition, real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.

## Recommended Next Step
G1.1 pressure-use hardening before broader G2
