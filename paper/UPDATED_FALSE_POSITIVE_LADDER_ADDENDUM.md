# Editable Behavior False Positive

## Definition

A model may respond correctly to explicit edit signals and support table-level edits, while failing to bind query behavior to support-inferred relations.

This matters because editability can look like relation internalization. A model may accept an edit signal, change a target output, and preserve locality, while still not using support examples to infer a relation regime or producing a stable causal relation subspace.

## Evidence

From the V1.2 interaction diagnostics for `edit_pressure_training`:

- `edit_state_swap_success = 1.000`
- `support_shuffle_drop = 0.000`
- `support_conditioned_accuracy = 0.500`
- `binding_sensitivity = 0.000`

V1.1 also shows that `edit_pressure_training` fails mainly on `relation_subspace_drop` in four out of five seeds.

## Conclusion

Edit-signal responsiveness is not sufficient evidence of relation internalization.

## Updated Ladder

- prediction success != relation internalization
- probe readability != relation internalization
- generic review text != relation internalization
- temporal memory != relation internalization
- blanket inspection != relation internalization
- edit-signal responsiveness != support-conditioned relation internalization
