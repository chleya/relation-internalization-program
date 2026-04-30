# B-Line PLOS Report

## 1. Research Question

Can a model form operational structure from continuous 2D dynamics without language labels, verified by behavior, structural intervention, and OOD gates?

## 2. W/O1/O2/L Framework

`W` is world process, `O1` is trajectory schema, `O2` is operational structure, and `L` is language overlay. This v1 tests `W -> O1 -> O2`, not `L -> L`.

## 3. Why Field-First Hybrid Schema Was Chosen

The main candidate starts from continuous fields and only then reads out sparse schemas. This avoids assuming objects first while retaining intervenable structure.

## 4. Why Behavior Alone Is Insufficient

Prediction, tracking, or inspection success can be produced by smoothness, memory, or saliency. PLOS requires behavior plus local structural intervention plus OOD generalization.

## 5. Environment Description

A 64x64 world with two moving balls, occlusion/crossing, collision/bounce, hidden local force field, and one budgeted 8x8 inspect action.

## 6. Model Branches And Roles

- `pixel_predictor`: first-order frame prediction baseline.
- `predictive_coding_model`: prediction-error substrate candidate.
- `trajectory_memory`: nearest-neighbor trajectory fragment baseline.
- `patch_graph_model`: local patch dynamic graph substrate candidate.
- `koopman_model`: low-rank dynamics substrate candidate.
- `world_model`: latent sequence predictor baseline.
- `slot_model`: object-biased second-order candidate.
- `field_model`: non-object-centric field-form candidate.
- `flow_checkpoint_model`: trajectory-flow checkpoint substrate candidate.
- `schema_model`: field-first hybrid schema main candidate.

## 7. Behavior Gate Results

| model | n | identity_after_occlusion | identity_after_crossing | event_boundary_alignment | intervention_sensitivity | relation_locality | noncausal_region_invariance | critical_region_selection_accuracy | inspection_value_gain | behavior_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| field_model | 3 | 0.746 | 0.792 | 0.142 | 1.000 | 0.075 | 0.009 | 0.069 | 0.069 | 0.363 |
| flow_checkpoint_model | 3 | 0.878 | 0.860 | 0.986 | 1.000 | 0.889 | 0.889 | 0.889 | 0.889 | 0.910 |
| koopman_model | 3 | 0.588 | 0.569 | 0.604 | 1.000 | 0.179 | 0.008 | 0.056 | 0.056 | 0.382 |
| patch_graph_model | 3 | 0.805 | 0.780 | 0.288 | 1.000 | 0.204 | 0.032 | 0.125 | 0.125 | 0.420 |
| pixel_predictor | 3 | 0.746 | 0.792 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.255 |
| predictive_coding_model | 3 | 0.746 | 0.792 | 0.167 | 1.000 | 0.197 | 0.026 | 0.097 | 0.097 | 0.390 |
| schema_model | 3 | 0.814 | 0.840 | 0.247 | 1.000 | 0.075 | 0.195 | 0.278 | 0.278 | 0.466 |
| slot_model | 3 | 0.814 | 0.840 | 0.257 | 0.600 | 0.000 | 0.319 | 0.319 | 0.319 | 0.434 |
| trajectory_memory | 3 | 0.746 | 0.792 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.255 |
| world_model | 3 | 0.746 | 0.792 | 0.000 | 1.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.380 |

## 8. Structural Intervention Results

| model | n | slot_causal_drop | slot_swap_consistency | event_latent_causal_drop | relation_edge_causal_drop | inspection_map_causal_drop | field_causal_drop | critical_field_locality | noncritical_field_invariance | structure_intervention_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| field_model | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.005 | 0.998 | 0.998 | 0.251 |
| flow_checkpoint_model | 3 | 0.000 | 0.000 | 0.044 | 0.044 | 0.004 | 0.001 | 0.999 | 0.999 | 0.262 |
| koopman_model | 3 | 0.000 | 0.000 | 0.033 | 0.000 | 0.112 | 0.003 | 0.999 | 0.999 | 0.268 |
| patch_graph_model | 3 | 0.000 | 0.000 | 0.000 | 0.004 | 0.098 | 0.004 | 0.998 | 0.998 | 0.263 |
| pixel_predictor | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| predictive_coding_model | 3 | 0.000 | 0.000 | 0.011 | 0.000 | 0.009 | 0.008 | 0.997 | 0.997 | 0.253 |
| schema_model | 3 | 0.016 | 0.016 | 0.013 | 0.013 | 0.005 | 0.003 | 0.999 | 0.999 | 0.258 |
| slot_model | 3 | 0.082 | 0.082 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.020 |
| trajectory_memory | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| world_model | 3 | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |

## 9. OOD Results

| model | n | ood_trajectory_generalization | ood_score |
| --- | --- | --- | --- |
| field_model | 3 | 0.436 | 0.436 |
| flow_checkpoint_model | 3 | 0.903 | 0.903 |
| koopman_model | 3 | 0.476 | 0.476 |
| patch_graph_model | 3 | 0.495 | 0.495 |
| pixel_predictor | 3 | 0.399 | 0.399 |
| predictive_coding_model | 3 | 0.448 | 0.448 |
| schema_model | 3 | 0.540 | 0.540 |
| slot_model | 3 | 0.552 | 0.552 |
| trajectory_memory | 3 | 0.399 | 0.399 |
| world_model | 3 | 0.399 | 0.399 |

## 10. Interpretation Of Each Branch

- `field_model` does not qualify. Field success would support field-form O if field perturbations produce local, predictable, task-aligned drops.
- `flow_checkpoint_model` qualifies. Success would support trajectory-checkpoint O, but it is discounted because checkpoint selection is an architectural prior.
- `koopman_model` does not qualify. Success would suggest low-rank dynamic modes can support operational structure, discounted by the linear-dynamics prior.
- `patch_graph_model` does not qualify. Success would suggest local patch dynamics can carry operational structure without object slots, discounted by the built-in grid prior.
- `pixel_predictor` does not qualify. If this branch has behavior without intervention evidence, interpret it as smoothness/prediction behavior, not operational structure.
- `predictive_coding_model` does not qualify. Success would suggest prediction-error dynamics can serve as an O1-to-O2 substrate, but only if the residual structures are local, causal, and OOD-stable.
- `schema_model` does not qualify. The schema model is the strongest candidate only if it passes behavior, intervention, and OOD gates together.
- `slot_model` does not qualify. Slot success would support an object-form O candidate only if slot interventions are local and task-aligned.
- `trajectory_memory` does not qualify. If this branch works in distribution but weakens on OOD, interpret it as trajectory memory rather than schema formation.
- `world_model` does not qualify. Good behavior from this branch is first-order latent prediction unless structure can be localized and intervened on.

## 11. Whether Any Model Qualifies As PLOS Candidate

Qualified PLOS candidates: flow_checkpoint_model.

Overall diagnostic summary:

| model | n | behavior_score | structure_intervention_score | ood_score | plos_candidate_score |
| --- | --- | --- | --- | --- | --- |
| field_model | 3 | 0.363 | 0.251 | 0.436 | 0.000 |
| flow_checkpoint_model | 3 | 0.910 | 0.262 | 0.903 | 0.434 |
| koopman_model | 3 | 0.382 | 0.268 | 0.476 | 0.000 |
| patch_graph_model | 3 | 0.420 | 0.263 | 0.495 | 0.000 |
| pixel_predictor | 3 | 0.255 | 0.000 | 0.399 | 0.000 |
| predictive_coding_model | 3 | 0.390 | 0.253 | 0.448 | 0.000 |
| schema_model | 3 | 0.466 | 0.258 | 0.540 | 0.000 |
| slot_model | 3 | 0.434 | 0.020 | 0.552 | 0.000 |
| trajectory_memory | 3 | 0.255 | 0.000 | 0.399 | 0.000 |
| world_model | 3 | 0.380 | 0.001 | 0.399 | 0.000 |

A model qualifies only when behavior, at least one relevant structural intervention gate, and OOD gates all pass. A zero score is not failure of the project; it means the current substrate did not satisfy the minimal evidence rule.

## 12. Substrate Audit Summary

Model inputs are observation-only: `past_frames`, `future_horizon`, `frame_size`, and `grid_size`. The substrate audit records architectural priors separately. Slot branches are discounted for object priors; field branches are discounted for field priors; schema branches are discounted for field plus schema-head priors. This audit does not prove blank-slate emergence, but it prevents treating injected structure as discovered structure.

The sweep also writes `results/null_control_summary.csv`. These blank/static controls flag branches that emit strong event or inspection structure when no dynamics are present.

Read `reports/B_LINE_SUBSTRATE_AUDIT.md` for the full branch-by-branch audit.

Important caveat: a qualifying score is still subject to structural audit. In this v1 diagnostic, some branches can score high on locality and invariance while causal-drop magnitudes remain small. Treat such a pass as a PLOS candidate, not as settled evidence of strong causal structure.

## 13. Claim Boundary

This is a toy diagnostic. It does not claim understanding, world-model intelligence, general physical reasoning, or language-free cognition.
