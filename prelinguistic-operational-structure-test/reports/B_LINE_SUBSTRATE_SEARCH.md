# B-Line Substrate Search

## Question

Can a less object-biased substrate reach the PLOS evidence rule: behavior + structural intervention + OOD?

## New Substrates

- `predictive_coding_model`: uses temporal prediction error and surprise fields as an O1 substrate.
- `patch_graph_model`: learns local patch-to-patch dynamics without object slots.
- `koopman_model`: learns a low-rank linear dynamic basis from observation-only frame pairs.
- `flow_checkpoint_model`: extracts continuity, anomaly, occlusion, and predicted-contact checkpoints from past frames.

These are not blank-slate substrates. They are candidate bases with explicit prior discounts recorded in `B_LINE_SUBSTRATE_AUDIT.md`.

## Result

At least one new substrate qualifies in this sweep: flow_checkpoint_model.

Caveat: `flow_checkpoint_model` is a PLOS candidate under the current v1 gates, but it has a high checkpoint-selection prior. Its causal-drop metrics remain small, so the result should be treated as a candidate foothold requiring hardening, not as a final internalization claim.

| model | n | behavior_score | structure_intervention_score | ood_score | plos_candidate_score |
| --- | --- | --- | --- | --- | --- |
| flow_checkpoint_model | 3 | 0.910 | 0.262 | 0.903 | 0.434 |
| koopman_model | 3 | 0.382 | 0.268 | 0.476 | 0.000 |
| patch_graph_model | 3 | 0.420 | 0.263 | 0.495 | 0.000 |
| predictive_coding_model | 3 | 0.390 | 0.253 | 0.448 | 0.000 |

## Null Controls

Blank/static-frame controls test whether a substrate emits structure without dynamics.

| model | n | null_event_activation | null_inspection_concentration | null_structure_applicable | null_prior_alarm |
| --- | --- | --- | --- | --- | --- |
| flow_checkpoint_model | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| koopman_model | 3 | 0.126 | 0.008 | 1.000 | 1.000 |
| patch_graph_model | 3 | 0.000 | 0.000 | 1.000 | 0.000 |
| predictive_coding_model | 3 | 0.000 | 0.000 | 1.000 | 0.000 |

## Interpretation

A zero PLOS score means the candidate did not pass the full gate. It does not mean the substrate is useless; it means current evidence is insufficient to call it pre-linguistic operational structure.
