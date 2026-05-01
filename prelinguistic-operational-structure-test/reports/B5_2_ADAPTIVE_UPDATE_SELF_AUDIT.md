# B5.2 Self-Audit

## What This Improves

- Directly attacks same-plan degeneracy.
- Tests whether inspection content changes trace update.
- Tests counterfactual inspection observations.
- Tests whether same initial observation with different information produces different plans.
- Tests contradictory feedback.
- Tests delayed feedback.
- Compares model update against scripted update.
- Compares model feedback revision against scripted feedback.
- Adds revision-specific ablation.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Still two-step closed loop.
- Inspection/feedback content is simulator-designed.
- Passing does not prove natural emergence or real active inference.
