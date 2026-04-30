# B3 Self-Audit

## What This Improves

- Moves from delayed trace representation to trace-guided active inspection.
- Tests budgeted inspect/action.
- Adds trace-vs-saliency conflict.
- Adds delayed information gain.
- Compares random, saliency, short-horizon, and oracle baselines.
- Tests inspection after private trace ablation.
- Keeps trace mechanism fixed from B2.3.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Inspect action is simplified.
- Oracle inspection value comes from simulator.
- Active inspection is not full control.
- Trace-guided inspection may still exploit generator regularities.
- B3 does not prove real-world engineering inspection.
- Passing B3 does not prove general active intelligence.
- Identical B3 scores do not prove independent mechanisms.

## False Positive Risks

- Model may select trace region without actual information gain.
- Saliency baseline may be too weak.
- Short-horizon baseline may be poorly implemented.
- Oracle information gain may encode evaluator bias.
- Trace ablation may damage unrelated model capacity.
- Inspect patch may reveal too much information.

## Required Failure Checks

1. random baseline passes
2. saliency baseline matches trace model
3. short-horizon baseline matches trace model
4. trace ablation does not reduce inspection performance
5. selected trace region does not improve delayed prediction
6. OOD delay inspection collapses
7. oracle score is low
