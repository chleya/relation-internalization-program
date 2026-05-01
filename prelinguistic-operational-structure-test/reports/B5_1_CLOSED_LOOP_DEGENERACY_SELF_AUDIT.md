# B5.1 Self-Audit

## What This Improves

- Investigates B5 high-score risk.
- Checks fixed closed-loop script.
- Checks inspect-always shortcut.
- Checks intervene-immediately shortcut.
- Checks closed-loop policy provenance.
- Checks trace-update specificity.
- Checks feedback-revision specificity.
- Checks epistemic/pragmatic value leakage.
- Checks planning-budget robustness.
- Checks baseline sanity.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Still only a two-step closed loop.
- Trace update and feedback revision may remain partly engineered.
- Budget stress may not reflect real compute constraints.
- Passing B5.1 does not prove real active inference.
- Passing B5.1 does not prove natural emergence.

## Required Failure Checks

1. three models choose same closed-loop plan per episode
2. model always inspects
3. model always intervenes immediately
4. update does not depend on inspection content
5. feedback revision does not depend on consequence
6. oracle/value leakage detected
7. planning budget stress collapses performance
8. inspect-always baseline matches model
9. intervene-immediately baseline matches model
10. trace ablation is nonspecific
