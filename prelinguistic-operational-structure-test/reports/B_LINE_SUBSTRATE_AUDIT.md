# B-Line Substrate Audit

## Purpose

This audit addresses whether a branch starts with operational structure already injected. It cannot prove a substrate is free of all priors. It records the designed priors and how much each result must be discounted.

## Input Firewall

All model `forward` and `intervene_structure` calls use an observation-only batch: `past_frames`, `future_horizon`, `frame_size`, and `grid_size`. Forbidden fields include `future_frames`, `ground_truth`, object IDs, event labels, relation labels, language, rules, and relation tables.

## Branch Audit

| model | role | observation_only_interface | language_or_relation_table_input | explicit_object_prior | explicit_field_prior | schema_head_prior | intervenable_structure_claim | prior_level | claim_discount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixel_predictor | first_order_baseline | True | False | False | False | False | False | low | Prediction behavior only; cannot count as operational structure without intervention evidence. |
| predictive_coding_model | prediction_error_substrate | True | False | False | False | False | True | medium_prediction_error_prior | Prediction-error structure is a temporal-compression prior; it must pass local intervention and OOD gates before counting as O. |
| trajectory_memory | memory_baseline | True | False | False | False | False | False | low_to_medium | Memory can fit fragments; OOD failure should be interpreted as non-structure. |
| patch_graph_model | local_patch_graph_substrate | True | False | False | False | False | True | medium_grid_prior | Patch graph success would support local operational structure, but grid locality is still supplied as a substrate prior. |
| koopman_model | low_rank_dynamics_substrate | True | False | False | False | False | True | medium_linear_dynamics_prior | Koopman success would support low-rank dynamic O only if mode interventions are local and OOD-stable. |
| world_model | latent_sequence_baseline | True | False | False | False | False | False | medium | Latent dynamics may help behavior, but structure is not localized by default. |
| slot_model | object_biased_candidate | True | False | True | False | False | True | high_object_prior | Slot success cannot prove object structure emerged; it tests whether object-form O is usable under an object-biased substrate. |
| field_model | field_form_candidate | True | False | False | True | False | True | medium_field_prior | Field success supports field-form O only if local field intervention and OOD gates pass. |
| flow_checkpoint_model | flow_checkpoint_substrate | True | False | False | False | True | True | high_checkpoint_prior | Flow-checkpoint success would show that continuity, anomaly, occlusion, and collision checkpoints are usable; it is discounted because checkpoint selection is an architectural prior. |
| schema_model | field_first_hybrid_schema_candidate | True | False | False | True | True | True | high_field_schema_prior | Schema success is strongest only when readouts are locally causal, task-aligned, and robust OOD. |

## Interpretation Rule

- Low-prior baselines can reject prediction and memory false positives, but they do not expose enough structure to qualify by behavior alone.
- Slot success is discounted because object structure is supplied as an architectural prior.
- Field success is not object-biased, but it is still a field prior rather than a blank substrate.
- Schema success would be strongest only if behavior, local structural intervention, and OOD gates pass together.
- A PLOS claim requires passing gates plus this audit; the audit itself is not evidence of operational structure.
- `results/null_control_summary.csv` is an additional blank/static-frame check for structure emitted without dynamics.
