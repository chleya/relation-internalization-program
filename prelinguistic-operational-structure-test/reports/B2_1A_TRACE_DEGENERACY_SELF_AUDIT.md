# B2.1a Self-Audit

## What This Improves

- Investigates identical B2.1 scores across three models.
- Checks per-attack and per-seed degeneracy.
- Checks predicted region distributions.
- Checks ground-truth leakage.
- Checks intervention applicability.
- Adds random-trace baseline.
- Adds oracle baseline.
- Adds trace-family and no-trace ablations.

## Remaining Weaknesses

- Still a toy 64x64 world.
- Audits are still implemented within the same codebase.
- Leakage checks may miss numerical leakage through derived fields.
- Oracle baseline may share evaluator assumptions.
- Random baseline may be too weak.
- Ablation may create OOD hidden states.
- Identical performance may still be genuine convergence under simple tasks.

## False Positive Risks

- Passing audit may not prove independent mechanisms.
- Trace-family ablation may hurt general capacity rather than trace specifically.
- No-trace ablation may be too destructive.
- Region distributions may differ even if models share same heuristic.
- No leakage in keys does not guarantee no statistical shortcut.

## Required Failure Checks

1. Random baseline high score.
2. Oracle baseline low score.
3. Ground-truth key leakage.
4. Interventions unsupported or fallback.
5. No drop under trace-family ablation.
6. No drop under no-trace ablation.
7. All attacks and seeds exactly identical without explanation.
