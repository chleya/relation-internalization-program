# B2 Self-Audit

## What This Improves

- Directly targets B1.1 delayed checkpoint failure.
- Tests delayed operational structure rather than short-horizon checkpoint saliency.
- Adds multi-delay and heldout-delay OOD.
- Adds causal trace intervention.
- Compares recurrent memory, field memory, and schema memory substrates.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Delayed episodes are hand-designed.
- Ground truth delays come from simulator.
- Memory architectures may inject delayed structure prior.
- Trace interventions may not capture distributed memory.
- Passing B2 does not prove natural emergence.

## False Positive Risks

- Model may infer delay from generator artifacts.
- Early saliency decoy may be too weak.
- Heldout delays may be interpolated rather than understood.
- Trace intervention may create OOD hidden states.
- Schema memory may pass because sparse delayed slots are built in.

## Required Failure Checks

1. flow_checkpoint remains short-horizon.
2. recurrent model passes behavior but fails trace intervention.
3. field memory passes OOD but fails causal trace.
4. schema memory passes because of hard-coded slot priority.
5. all models fail heldout delays.
6. delayed score high but early saliency rejection low.
