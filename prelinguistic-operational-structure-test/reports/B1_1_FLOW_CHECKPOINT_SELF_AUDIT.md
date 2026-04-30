# B1.1 Self-Audit

## What This Improves

- Attacks the only PLOS candidate.
- Tests dynamic decoy robustness.
- Tests delayed causality.
- Tests competing checkpoint selection.
- Tests relocation OOD.
- Compares causal deletion vs visual deletion.
- Tests anti-prior worlds.

## Remaining Weaknesses

- Still a toy 64x64 world.
- flow_checkpoint_model still contains strong priors.
- Hardening attacks are hand-designed.
- Passing does not imply blank-slate emergence.
- Causal deletion may create out-of-distribution inputs.
- Endpoint shift may under-measure relational changes.
- Competing checkpoint oracle may encode evaluator assumptions.

## False Positive Risks

- Model may exploit attack generator regularities.
- Delayed checkpoint may still be visually inferable.
- Decoys may be too weak.
- Anti-prior episodes may be too narrow.
- Causal deletion may affect visual continuity rather than causal structure.

## Required Failure Checks

1. dynamic decoy moves selection
2. delayed checkpoint fails
3. fixed priority wins over information value
4. relocation OOD collapses
5. causal deletion not stronger than visual deletion
6. anti-prior trap selected
7. hardening pass with tiny causal endpoint shift
