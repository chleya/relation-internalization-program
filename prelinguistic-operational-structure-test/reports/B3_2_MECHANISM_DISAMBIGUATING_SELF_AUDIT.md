# B3.2 Self-Audit

## What This Improves

- Addresses B3.1 same-region degeneracy.
- Creates family-specific inspect targets.
- Separates recurrent / field / schema inspection values.
- Adds non-linguistic goal conditioning.
- Adds mechanism disagreement episodes.
- Adds family-specific trace ablation.
- Tests whether models can switch inspect regions by goal family.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Family-specific targets are synthetic.
- Goal code is hand-defined.
- Value decomposition is evaluator-designed.
- Mechanism disagreement episodes may encode evaluator assumptions.
- Passing does not prove real-world active inspection.
- Passing does not prove complete mechanism independence.

## False Positive Risks

- Models may use goal-code switching rather than learned mechanism-specific inspection.
- Family-specific value maps may be too easy.
- Ablation may damage general capacity rather than family trace.
- Different inspect regions may come from task construction rather than natural mechanism separation.
- Oracle family values may reflect evaluator bias.

## Required Failure Checks

1. family-specific regions collapse to same region
2. task-conditioned switching fails
3. cross-model same-region rate remains high
4. trace ablation is not family-specific
5. saliency or short-horizon baseline matches model
6. oracle family inspection score is low
7. model succeeds by goal-code shortcut without using trace
