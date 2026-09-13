# G1 Minimal Generator Results

## Summary
G1 runs a compact rule-search generator that induces an actionability/update mask from interaction history rather than receiving the B-line mask directly.

## Metrics
- g1_generator_mean_score = 0.890
- g1_test_score = 0.892
- g1_ood_score = 0.868
- g1_ood_gain_over_random = 0.637
- g1_ood_gain_over_hand_designed = 0.273
- g1_oracle_gap = 0.110
- g1_mask_f1 = 0.934
- g1_compression_cost = 0.500

## Interpretation
- beats_random_on_ood: true
- beats_hand_designed_on_ood: true
G1 is useful only if the generated rule improves over weak baselines while preserving an explicit oracle gap and failure report.

## Claim Boundary
G1 is a minimal toy generator experiment. It does not prove autonomous cognition, real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.

## Next Step
G2 should test whether the generated rule family can become less hand-scaffolded, for example by inducing feature combinations or update primitives rather than selecting thresholds over a fixed vocabulary.
