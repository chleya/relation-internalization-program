# Auto Report: Neural Relation Probe

Date: 2026-04-28

## Main Question

Does a small neural classifier encode and causally use the true `texture/wet -> resource` relation?

## Tests

- OOD accuracy
- spurious attack accuracy
- relation probe selectivity over random-label control
- relation-subspace intervention drop
- nuisance-subspace intervention drop
- gated neural relation score

## Mode Comparison

Mean over seeds 0-4:

```text
mode      ood   spurious  selectivity  relation_drop  nuisance_drop  gated
base      1.000 1.000     0.793        0.483          0.000          0.819
shortcut  0.569 0.111     0.749        0.441          0.441          0.000
```

## Interpretation

The base model passes because it remains robust under OOD and spurious attack, and relation-subspace removal damages behavior more than nuisance-subspace removal.

The shortcut model fails despite readable relation information. It performs poorly under spurious attack and relation-subspace removal is not more specific than nuisance-subspace removal.

This preserves the main line:

```text
relation internalization requires usable, transferable, intervention-relevant structure;
probe readability alone is not enough.
```

## Current Claim Boundary

Supported:

- In this toy setting, a neural hidden state can contain linearly readable relation information.
- Subspace removal can test whether that relation information is behaviorally relevant.
- The gate rejects a shortcut model that appears probe-readable but fails transfer and specificity.

Not supported:

- General spontaneous relation discovery.
- Real engineering deployment.
- Claims about large language models.

## v0.4 Neural-to-Table Bridge

Added:

```text
MLP hidden state + relation probe -> extracted editable relation table
```

The extracted table supports:

- `infer_resource(context)`
- `act(context)`
- `edit_rule(condition, new_outcome)`
- `describe_relations()`

Mean over seeds 0-4:

```text
base:
table_ood_accuracy: 1.000
table_spurious_attack_accuracy: 1.000
table_relation_alignment: 1.000
gated_extraction_score: 1.000
edit_success: true

shortcut:
table_ood_accuracy: 0.455
table_spurious_attack_accuracy: 0.456
table_relation_alignment: 0.333
gated_extraction_score: 0.000
edit_success: true
```

Interpretation:

The bridge works when the neural model has a stable relation representation. It also exposes failure when the model learned a shortcut: the extracted table becomes editable, but its contents are wrong.

This is the first minimal implementation of:

```text
neural representation -> explicit relation table -> editable action policy
```

## v0.5 Extraction Gate

Added `gated_extraction_score`.

The score is zero unless:

```text
table_ood_accuracy >= 0.9
table_spurious_attack_accuracy >= 0.9
table_relation_alignment >= 0.9
edit_success == true
```

This prevents a false positive where a wrong extracted table is still technically editable.
