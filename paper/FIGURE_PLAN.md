# Figure Plan

## Figure 1: Evidence Ladder

Pipeline:

```text
Static relation -> Neural probe -> Slope relation chain -> Temporal delay -> Partial observability -> Active inspection
```

Under each node, show the false positive eliminated:

| Node | False positive eliminated |
| --- | --- |
| Static relation | Memory, fitting, prediction-only behavior, shortcut policies, predefined-link dependence |
| Neural probe | Probe readability without causal or behavioral specificity |
| Slope relation chain | Surface labels, structural memory, generic review language |
| Temporal delay | Same-step logic, fixed-delay templates, temporal memory without editable links |
| Partial observability | Relation discovery without uncertainty handling |
| Active inspection | Blanket inspection and simple field-selection heuristics |

Purpose:
Show the paper as a diagnostic ladder rather than a project log.

## Figure 2: Accuracy vs Gated Relation Score

Show examples where ordinary success is high but gated score is zero:

- `structural_memory_temporal`;
- `discovery_relation_agent` in R2;
- shortcut neural model;
- `structural_memory`.

Recommended design:

- x-axis: ordinary metric, such as OOD success, temporal prediction, partial observation success, or probe readability.
- y-axis: gated relation score.
- Mark positive agents and false-positive baselines with different colors.

Purpose:
High performance is not relation internalization.

## Figure 3: Edit / Audit / Counterfactual Requirements

Show relation-chain agents versus memory/generic/shortcut controls on:

- edit success;
- audit score;
- counterfactual accuracy;
- relation recovery or relation alignment.

Recommended sources:

- food-world static relation results;
- slope-relation-toy results;
- temporal V2/V2.1 results;
- R1/R1.1/R1.2 results.

Purpose:
Make visible that the diagnostic is about usable, editable, auditable structure rather than output success alone.

## Figure 4: Uncertainty and Inspection Progression

R2 -> R2.1 -> R3:

- R2: discovery-only fails uncertainty.
- R2.1: missing-always fails precision/cost.
- R3: active inspection passes target selection.

Recommended panels:

- R2: partial observation success versus gated score for `discovery_relation_agent` and `uncertainty_discovery_agent`.
- R2.1: inspection precision and unnecessary inspection rate for `missing_always_inspect` and `relation_specific_uncertainty_agent`.
- R3: inspection target accuracy, information gain efficiency, and budgeted safe action rate for active inspection versus simple baselines.

Purpose:
Show the progression from relation discovery to uncertainty recognition to cost-aware active inspection.
