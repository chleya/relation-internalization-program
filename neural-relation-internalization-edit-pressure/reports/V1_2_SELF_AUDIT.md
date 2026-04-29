# V1.2 Self-Audit

## What V1.2 improves

- Tests whether edit-pressure relation structure may be nonlinear.
- Tests whether model uses support-conditioned relation state.
- Tests whether edit state has causal effect.
- Tests whether query-support binding exists.
- Extends intervention beyond linear subspace removal.

## Remaining weaknesses

- Toy world remains simple.
- Probes are still diagnostic, not proof of understanding.
- Nonlinear probes can overfit small datasets.
- Support shuffle may disrupt distribution in artificial ways.
- Edit-state swap requires model-specific hooks.
- Interaction evidence score is not a replacement for gated internalization score.
- No claim of general causal representation.

## False positive risks

- MLP probe may decode information that the policy does not use.
- Swap tests may exploit architecture artifacts.
- Ablation may cause out-of-distribution hidden states.
- Support-conditioned behavior may still be shallow memorization.

## Required failure checks

1. MLP probe high but support shuffle low.
2. Edit-state swap has no behavioral effect.
3. Query-support binding fails.
4. Ablation effects are only present under OOD-hidden artifacts.
5. Interaction evidence is strong only for one seed.
