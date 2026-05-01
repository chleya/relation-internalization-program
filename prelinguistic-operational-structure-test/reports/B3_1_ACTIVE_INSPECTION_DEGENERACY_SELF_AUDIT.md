# B3.1 Active Inspection Degeneracy Self-Audit

## What This Improves

- Checks per-episode inspect-region overlap.
- Adds inspection policy provenance.
- Compares trace-family-specific inspection scorers.
- Disables any shared inspection policy path.
- Adds disagreement-inspection episodes.
- Rechecks random, saliency, short-horizon, and oracle baselines.
- Tests trace-ablation specificity.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Audit episodes are synthetic.
- Provenance can miss hidden shared low-level features.
- High overlap can be caused by a simple task rather than literal shared code.
- Passing does not prove independent mechanisms.

## Required Failure Checks

1. cross-model inspect regions match too often
2. shared inspection policy is used
3. private inspection scorer is not used
4. inspection scorer correlation is too high
5. disagreement-inspection episodes do not diverge
6. random/saliency/short-horizon baselines are too strong
7. private trace ablation is not specific
