# README: Neural Stage Update

## What Changed After V1.1 and V1.2

The neural stage is now frozen as a mixed but useful result.

V1.1 localized the `edit_pressure_training` failure. The model passes many behavioral and table-level metrics, but fails mainly on the `relation_subspace_drop` gate in four out of five seeds. This shows that table-level editability and stable causal relation subspaces are separable diagnostics.

V1.2 tested whether the missing relation evidence might instead be nonlinear or interaction-mediated. The result does not support a strong interaction-mediated relation claim. The edit pathway has behavioral effect (`edit_state_swap_success = 1.000`), but support-conditioned binding remains weak (`support_shuffle_drop = 0.000`, `support_conditioned_accuracy = 0.500`, `binding_sensitivity = 0.000`).

## Why Edit-Pressure Should Not Be Described as Stable Success

`edit_pressure_training` should not be described as a stable neural relation-internalization result. It shows table-level editability and edit-state responsiveness, but it does not show robust support-conditioned relation binding or stable causal relation subspaces.

The safe wording is:

```text
Edit-pressure training is mixed: it produces a behaviorally effective edit pathway, but not robust support-conditioned relation internalization.
```

## How to Cite the Neural Stage in the Main Paper

Use the neural stage to support two claims:

1. Counterfactual pressure is the strongest current non-handwritten neural positive result in the toy setting.
2. Editable behavior is itself a false positive unless paired with support-conditioned binding and causal representation evidence.

Do not use the neural stage as proof that edit-pressure solves relation internalization.

## Results to Use

Main sweep:

- `pure_prediction`: gated score `0.000`
- `prediction_bottleneck`: gated score `0.000`
- `counterfactual_training`: gated score about `0.981`
- `edit_pressure_training`: gated score about `0.200`
- `explicit_table_oracle`: gated score `1.000`

V1.1:

- `edit_pressure_training` fails mainly on `relation_subspace_drop` in 4/5 seeds.
- Multi-site drops are weak at `query_hidden` and `combined_hidden`, and near zero at `support_relation_state` and `post_edit_state`.

V1.2:

- `edit_state_swap_success = 1.000`
- `support_shuffle_drop = 0.000`
- `support_conditioned_accuracy = 0.500`
- `binding_sensitivity = 0.000`
- no extra nonlinear relation gain from nonlinear probes.

## What Not to Claim

- Do not claim edit-pressure has solved neural relation internalization.
- Do not claim the model has general causal understanding.
- Do not claim the result applies to large language models.
- Do not claim the result applies to real engineering systems.
- Do not claim nonlinear relation structure is proven.
- Do not claim support-conditioned binding has been achieved.

The neural stage strengthens the paper not by showing that edit-pressure succeeds, but by showing that even editability can be a false positive unless it is paired with support-conditioned binding and causal representation evidence.
