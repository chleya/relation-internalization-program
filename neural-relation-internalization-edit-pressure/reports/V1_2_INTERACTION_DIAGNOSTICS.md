# V1.2 Nonlinear / Interaction Causal Check

## 1. Motivation

V1.1 found that `edit_pressure_training` has extracted-table editability but unstable `relation_subspace_drop`. V1.2 asks whether the relation structure is absent, or whether it is encoded in a nonlinear, interaction-mediated, episode-conditioned form that linear probe-subspace intervention can miss.

## 2. Diagnostics Added

- nonlinear probes
- support-state shuffle
- edit-state swap
- query-support binding
- multi-site ablation matrix

## 3. Results

### Interaction Summary

| model | nonlinear gain | support shuffle | edit swap | binding acc | support ablation | interaction evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| counterfactual_training | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| edit_pressure_training | 0.000 | 0.000 | 1.000 | 0.500 | 0.001 | 0.351 |
| pure_prediction | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Observed diagnostic pattern:

The observed edit-pressure pattern is weak support-query interaction evidence: support shuffle has little effect and same-query support binding is not reliable. Edit-state swap is strong, so the edit pathway has behavioral effect even when support-conditioned regime use remains weak.

### Nonlinear Probe Comparison

| model | linear relation | MLP relation | tree relation | linear nuisance | MLP nuisance | tree nuisance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| counterfactual_training | 1.000 | 1.000 | 0.925 | 1.000 | 1.000 | 0.971 |
| edit_pressure_training | 1.000 | 1.000 | 0.899 | 1.000 | 1.000 | 0.994 |
| pure_prediction | 1.000 | 1.000 | 0.899 | 1.000 | 1.000 | 0.974 |

### Support Shuffle

| model | normal | shuffled | drop |
| --- | ---: | ---: | ---: |
| counterfactual_training | 0.000 | 0.000 | 0.000 |
| edit_pressure_training | 0.667 | 0.667 | 0.000 |
| pure_prediction | 0.000 | 0.000 | 0.000 |

### Edit State Swap

| model | normal edit acc | swapped effect | swapped locality | success |
| --- | ---: | ---: | ---: | ---: |
| counterfactual_training | 0.000 | 0.000 | 0.000 | 0.000 |
| edit_pressure_training | 1.000 | 1.000 | 1.000 | 1.000 |
| pure_prediction | 0.000 | 0.000 | 0.000 | 0.000 |

### Binding Test

| model | support-conditioned acc | binding sensitivity | invariance failure |
| --- | ---: | ---: | ---: |
| counterfactual_training | 0.000 | 0.000 | 0.000 |
| edit_pressure_training | 0.500 | 0.000 | 1.000 |
| pure_prediction | 0.000 | 0.000 | 0.000 |

### Ablation Matrix Summary

| model | site | max drop | mean drop |
| --- | --- | ---: | ---: |
| counterfactual_training | combined_hidden | 0.825 | 0.473 |
| counterfactual_training | query_embedding | 0.825 | 0.473 |
| edit_pressure_training | combined_hidden | 0.835 | 0.474 |
| edit_pressure_training | edit_embedding | 0.000 | 0.000 |
| edit_pressure_training | post_edit_state | 0.020 | 0.001 |
| edit_pressure_training | query_embedding | 0.740 | 0.420 |
| edit_pressure_training | support_relation_state | 0.005 | 0.000 |
| pure_prediction | combined_hidden | 0.735 | 0.383 |
| pure_prediction | query_embedding | 0.735 | 0.383 |

## 4. Interpretation

A. Strong nonlinear/interaction evidence would mean edit-pressure relation use may exist outside a single linear subspace.

B. Weak nonlinear/interaction evidence would mean edit-pressure mainly learned shallow editable patterns rather than strong support/edit-state causal use.

C. Seed-specific evidence would mean the training pressure can sometimes form relation structure, but the mechanism is unstable across seeds.

## 5. Relation to V1.1

This does not overturn the original gated result. It explains why `edit_pressure_training` may fail linear relation-subspace diagnostics despite table-level editability. The diagnostic `interaction_evidence_score` is not a replacement for `gated_internalization_score`.

## 6. Claim Boundary

Supported:
- diagnostic localization of edit-pressure representation form in a toy setting.

Unsupported:
- proof of general neural relation internalization.
- proof of nonlinear causal representation.
- large model claims.
- real-world causal discovery.
