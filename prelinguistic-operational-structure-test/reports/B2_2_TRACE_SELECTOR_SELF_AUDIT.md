# B2.2 Self-Audit

## What This Improves

- Addresses B2.1a identical prediction degeneracy.
- Tracks trace provenance.
- Adds selector-free variants.
- Adds trace disagreement episodes.
- Adds source-specific ablations.
- Adds independent trace scorer comparison.
- Adds shared-selector ablation.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Selector-free variants may be weaker by construction.
- Provenance reporting can be incomplete or misleading.
- Disagreement episodes are hand-designed.
- Trace scorer correlation may be high because task is simple.
- Source-specific ablation may create OOD hidden states.
- Passing does not prove natural emergence.

## False Positive Risks

- Model-private scorer may still reuse shared features.
- Selector-free variants may retain hidden shared heuristic.
- Disagreement episodes may encode evaluator assumptions.
- Shared selector usage may be underreported.
- Ablation may harm unrelated capacity.
- Low prediction overlap may come from noise, not mechanism separation.

## Required Failure Checks

1. shared selector usage remains high.
2. selector-free variants collapse.
3. trace disagreement episodes do not produce divergence.
4. model-private trace ablation has small effect.
5. shared selector ablation has larger effect than private trace ablation.
6. trace scorers are nearly identical.
7. provenance fields are missing or unknown.
