# Neural Relation Internalization under Counterfactual and Edit Pressure

## Motivation

The earlier relation-agent diagnostics used explicit relation machinery: relation tables, editable links, audits, and inspection policies. Those stages were useful for defining what relation internalization should require, but they left open a stronger neural question: can a model without a hand-written positive relation table produce relation-internalization-like behavior under training pressure?

This stage answers that question conservatively. It tests prediction, compression, counterfactual, and edit-pressure training in the same toy food/poison world, and evaluates them with transfer, shortcut rejection, reversal adaptation, neural-to-table extraction, table edit, edit locality, and causal subspace diagnostics.

## Experimental Setup

The environment is a food/poison/neutral relation world with four input variables:

- `texture`: A/B/C
- `wet`: dry/wet
- `color`: red/blue
- `odor`: strong/weak

The true relation variables are `texture` and `wet`. The nuisance shortcut variables are `color` and `odor`. During training, shortcut variables are correlated with resource labels. During OOD and spurious-attack evaluation, these shortcuts are randomized or reversed.

The base relation maps `A,dry` to food, `A,wet` to poison, `B` to poison, and `C` to neutral. A reversal regime swaps the food/poison relation for `A` and maps `B` to food. The tests ask whether a model follows the relation variables rather than the shortcut variables, adapts to relation reversal, and yields extractable/editable relation behavior.

## Models

- `pure_prediction`: a supervised MLP trained only on resource prediction.
- `prediction_bottleneck`: a supervised MLP with a small bottleneck representation.
- `counterfactual_training`: an MLP trained with nuisance-invariance and relation-counterfactual pressure.
- `edit_pressure_training`: a support-conditioned neural model trained on edit episodes.
- `explicit_table_oracle`: a hand-written relation table upper bound, not neural evidence.

## Results

| model | gated_internalization_score | interpretation |
| --- | ---: | --- |
| `pure_prediction` | 0.000 | Prediction alone fails the gate. |
| `prediction_bottleneck` | 0.000 | Compression does not isolate relation structure. |
| `counterfactual_training` | ~0.981 | Strongest non-handwritten neural positive result. |
| `edit_pressure_training` | ~0.200 | Mixed: editable behavior without stable causal relation evidence. |
| `explicit_table_oracle` | 1.000 | Hand-written upper bound. |

Counterfactual training is currently the strongest non-handwritten neural positive result, while edit-pressure training reveals that edit responsiveness can be mistaken for relation internalization.

## Failure Localization of Edit-Pressure

V1.1 localizes the edit-pressure failure to the relation-subspace gate. Four out of five seeds fail mainly on `relation_subspace_drop`; only seed 2 clearly passes.

| seed | relation_subspace_drop | gate |
| --- | ---: | --- |
| 0 | 0.188 | fail |
| 1 | 0.138 | fail |
| 2 | 0.528 | pass |
| 3 | 0.144 | fail |
| 4 | 0.176 | fail |

Multi-site intervention shows weak drops at `query_hidden` and `combined_hidden`, and near-zero drops at `support_relation_state` and `post_edit_state`:

| site | mean relation_drop |
| --- | ---: |
| `query_hidden` | ~0.140 |
| `support_relation_state` | ~0.000 |
| `combined_hidden` | ~0.148 |
| `post_edit_state` | ~0.000 |

The failure is not table extraction or local table edit. Those are strong. The failure is stable causal relation evidence.

## Nonlinear / Interaction Diagnostics

V1.2 asks whether edit-pressure relation use exists in nonlinear or support-query-edit interaction form. The result is mixed but mostly negative for strong interaction-mediated relation structure:

- `interaction_evidence_score ≈ 0.351`
- `edit_state_swap_success = 1.000`
- `support_shuffle_drop = 0.000`
- `support_conditioned_accuracy = 0.500`
- `binding_sensitivity = 0.000`
- nonlinear probes do not show extra nonlinear relation gain; linear and MLP probes are already readable.

The edit pathway has behavioral effect. However, support-query relation binding is weak, so the result does not support the claim that edit-pressure has produced robust support-conditioned relation internalization.

## Interpretation

Counterfactual training is currently the strongest non-handwritten neural positive result, while edit-pressure training reveals that edit responsiveness can be mistaken for relation internalization.

The important separation is between:

1. table-level editability;
2. edit-state responsiveness;
3. support-conditioned relation binding;
4. stable causal relation subspace.

The edit-pressure model shows the first two more strongly than the latter two. Therefore, it should not be described as a stable success.

## Implication for the Main Paper

This stage adds a new false positive to the diagnostic ladder:

```text
editable behavior is not necessarily relation internalization.
```

The neural stage strengthens the paper not by showing that edit-pressure succeeds, but by showing that even editability can be a false positive unless it is paired with support-conditioned binding and causal representation evidence.
