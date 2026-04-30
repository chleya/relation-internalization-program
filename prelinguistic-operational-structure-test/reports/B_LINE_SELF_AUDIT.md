# B-Line PLOS Self-Audit

## What This Improves

- Tests `W -> O1 -> O2` rather than `L -> L`.
- Separates behavior success from structure evidence.
- Adds structural intervention gates.
- Adds field-based branches to avoid object-centric bias.
- Keeps language out of v1.
- Adds an observation-only model input firewall.
- Adds a substrate audit for injected object, field, and schema priors.

## Remaining Weaknesses

- Toy world is simple.
- Ground truth comes from simulator.
- Auxiliary heads may bias structure.
- Structural interventions may miss distributed representations.
- Field model may secretly encode object-like structure.
- Slot model has built-in object bias.
- Field and schema branches still contain architectural priors, not blank-slate emergence.
- Flow-checkpoint success is high-prior checkpoint structure, not blank-slate emergence.
- Current intervention metrics can under-measure sparse local causal effects and over-credit locality/invariance.
- Passing does not imply real-world physical intelligence.

## False Positive Risks

- Pixel predictor may exploit smoothness.
- Trajectory memory may overfit common fragments.
- Slot model may pass because objects are built in.
- Field model may pass because force field is too simple.
- Inspection policy may learn saliency rather than information value.
- Intervention may create out-of-distribution hidden states.

## Required Failure Checks

1. Behavior high but structure intervention low.
2. Structure intervention high but behavior low.
3. OOD collapse.
4. Pixel predictor passes all gates.
5. Slot-only success with field failure.
6. Field-only success with slot failure.
7. Schema model succeeds only through one brittle head.
