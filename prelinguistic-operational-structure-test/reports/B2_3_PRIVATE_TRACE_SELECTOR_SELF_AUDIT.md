# B2.3 Self-Audit

## What This Improves

- Directly addresses B2.2 shared selector failure.
- Removes shared selector path from recurrent / field / schema models.
- Requires model-private trace scoring.
- Requires provenance proof.
- Re-runs B2 and B2.1 under private selectors.
- Re-runs B2.1a/B2.2-style audits.
- Tests source-specific ablation.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Private selectors are still hand-designed.
- Selector-free variants may be weaker by construction.
- Provenance reporting may be incomplete.
- Private scorers may still share low-level features.
- Disagreement episodes are synthetic.
- Passing does not prove natural emergence.

## False Positive Risks

- Private selector may secretly reproduce shared heuristic.
- Trace family specificity may be inflated by scorer normalization.
- Ablation may damage general capacity, not trace specifically.
- Disagreement divergence may come from noise, not mechanism separation.
- B2/B2.1 regression may be easier than real delayed causality.

## Required Failure Checks

1. shared selector usage > 0
2. fallback usage high
3. private scorer not used
4. B2 regression fails
5. B2.1 regression fails
6. prediction overlap remains too high
7. disagreement divergence too low
8. private trace ablation drop too low
9. shared selector ablation still dominates
