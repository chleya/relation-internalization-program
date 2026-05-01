# B4 Self-Audit

## What This Improves

- Moves from inspection to local intervention/action selection.
- Keeps action space minimal.
- Tests intervention vs inspection distinction.
- Adds wrong-region intervention penalty.
- Adds family-specific intervention targets.
- Adds trace-ablation action test.
- Adds random/saliency/short-horizon/inspect-only/oracle baselines.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Actions are simplified.
- Intervention values are simulator-defined.
- No continuous control.
- No real robot or real engineering environment.
- Family-specific intervention targets are synthetic.
- Passing B4 does not imply deployable intervention intelligence.

## False Positive Risks

- Action type may be inferred from episode type shortcut or model-family priors.
- Region may be selected correctly but action type may be trivial.
- Inspect-only baseline may be too weak.
- Oracle intervention values may encode evaluator bias.
- Wrong-region penalty may be too easy.
- Trace ablation may damage general capacity, not action selection specifically.
- Family-specific targets may leak through synthetic generator regularities.

## Required Failure Checks

1. random baseline passes
2. saliency baseline matches trace model
3. short-horizon baseline matches trace model
4. inspect-only matches intervention
5. wrong-region intervention is not penalized
6. trace ablation does not reduce action performance
7. oracle intervention score is low
8. action type accuracy is low despite region accuracy
