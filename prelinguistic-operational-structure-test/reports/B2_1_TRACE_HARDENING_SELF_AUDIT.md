# B2.1 Self-Audit

## What This Improves

- Tests whether trace-bearing models actually use causal trace.
- Adds false delayed trace attacks.
- Adds trace swap.
- Adds trace deletion specificity.
- Adds multi-source trace conflict.
- Adds noisy trace robustness.
- Adds trace length extrapolation.
- Adds trace compression pressure.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Trace attacks are hand-designed.
- Ground truth trace comes from simulator.
- Models may exploit generator artifacts.
- Trace interventions may create OOD hidden states.
- Compression pressure may not match natural resource limits.
- Passing does not prove blank-slate emergence.

## False Positive Risks

- Model rejects false trace using superficial difference.
- Trace swap creates unnatural hidden state.
- Deletion affects visual continuity rather than trace.
- Conflict oracle encodes evaluator bias.
- Noisy trace remains too clean.
- Length extrapolation may still be interpolation in disguise.
- Compression keeps causal trace only because architecture privileges it.

## Required Failure Checks

1. false trace selected
2. trace swap has no behavioral effect
3. true trace deletion no stronger than non-trace deletion
4. wrong trace source wins under conflict
5. noisy trace causes collapse
6. delay=8/10 extrapolation fails
7. compression preserves saliency over causal trace
