# B4.2 Self-Audit

## What This Improves

- Directly addresses fixed_action_type_rate = 1.000.
- Tests action-type-specific intervention values.
- Penalizes correct-region-wrong-action.
- Adds family-action mapping stress.
- Adds action-type counterfactuals.
- Adds fixed-action baseline.
- Adds action-type ablation.
- Adds action-type OOD.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Action values are simulator-defined.
- Action-type mapping is hand-designed.
- No continuous control.
- No real robot or real engineering environment.
- Passing does not imply real intervention intelligence.

## False Positive Risks

- Models may learn episode-type to action-type shortcut.
- Family-action mapping may be too explicit.
- OOD action mapping may still be predictable from generator artifacts.
- Action-type counterfactual may encode evaluator assumptions.
- Action-type ablation may disturb unrelated action capacity.
- Fixed-action baseline may be too weak.

## Required Failure Checks

1. fixed_action_type_rate remains high
2. correct-region-wrong-action not penalized
3. fixed-action baseline matches model
4. action-type ablation does not change action type
5. action-type OOD fails
6. oracle action-type baseline low
7. action type succeeds by shortcut without trace
