# Contributions V1

## Contribution 1: A Diagnostic Definition of Relation Internalization

The paper defines relation internalization operationally:

```text
External relations become internal usable structures.
```

Usable means that the structure supports:

- transfer;
- counterfactual action;
- local edits;
- audits;
- temporal indexing;
- uncertainty recognition;
- cost-aware inspection.

This is not presented as a general theory of intelligence. It is a diagnostic standard for controlled toy environments.

## Contribution 2: A False-Positive Elimination Ladder

The paper's main methodological contribution is a ladder of negative controls. Each rung identifies a behavior that can look relation-like under ordinary metrics but fails structural gates.

False positives include:

- prediction success;
- bottleneck compression;
- probe readability;
- structural memory;
- generic review text;
- temporal memory;
- fixed-delay templates;
- relation discovery without uncertainty;
- blanket inspection;
- first-missing/random/risk-first inspection;
- edit-signal responsiveness without support-conditioned binding.

The ladder makes the central argument concrete: relation-internalization claims require more than ordinary task performance.

## Contribution 3: Neural Diagnostics Without a Hand-Written Positive Table

Earlier R-series positive agents intentionally used explicit relation machinery. The neural stage tests a weaker but important question: can relation-internalization-like behavior appear without a hand-written positive relation table?

Current result:

- `pure_prediction`: gated score `0.000`;
- `prediction_bottleneck`: gated score `0.000`;
- `counterfactual_training`: gated score about `0.981`;
- `edit_pressure_training`: gated score about `0.200`;
- `explicit_table_oracle`: gated score `1.000`.

The conservative conclusion is that counterfactual training is the strongest current non-handwritten positive condition in the toy setting. This does not show spontaneous relation emergence or general causal discovery.

## Contribution 4: Editability as a Newly Identified False Positive

The edit-pressure result adds a sharper diagnostic distinction:

1. table-level editability;
2. edit-state responsiveness;
3. support-conditioned relation binding;
4. stable causal relation subspace.

`edit_pressure_training` shows table-level editability and edit-state responsiveness, but weak support-conditioned binding and unstable causal relation-subspace evidence. V1.2 reports:

- `edit_state_swap_success = 1.000`;
- `support_shuffle_drop = 0.000`;
- `support_conditioned_accuracy = 0.500`;
- `binding_sensitivity = 0.000`.

Therefore, editable behavior is not sufficient evidence of relation internalization.

## Contribution 5: A Conservative Claim Boundary

The paper explicitly rules out overclaims:

- no real slope monitoring;
- no real engineering safety;
- no deployment claim;
- no general causal discovery;
- no object permanence claim;
- no LLM replacement claim;
- no claim that large neural systems naturally internalize relations.

This boundary is part of the contribution. The paper is useful because it makes weaker evidence harder to overinterpret.
