# B-Line Flow-Checkpoint Hardening

## Purpose

This hardening pass tests whether the current PLOS candidate is merely exploiting checkpoint priors, visible decoy patches, or weak intervention metrics.

## Result

flow_checkpoint_model survives this hardening pass.

| candidate_gate_preserved | decoy_patch_stability | static_decoy_suppression | causal_endpoint_shift | hardening_score |
| --- | --- | --- | --- | --- |
| 1.000 | 0.833 | 1.000 | 0.250 | 0.771 |

## Checks

- `candidate_gate_preserved`: the base PLOS candidate score remains nonzero.
- `decoy_patch_stability`: adding a noncausal faint rectangle does not move the selected checkpoint.
- `static_decoy_suppression`: static objects plus a decoy patch do not trigger operational structure.
- `causal_endpoint_shift`: ablating the selected event/checkpoint changes future endpoints in causal episodes.

## Boundary

Passing this hardening pass still does not prove blank-slate emergence. The candidate remains a high-prior checkpoint substrate and needs further attacks.
