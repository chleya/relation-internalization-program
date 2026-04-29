# Methods and Diagnostic Criteria V1

## Diagnostic Framing

The paper treats relation internalization as an operational diagnostic claim. The gates are not intended as a general theory of intelligence. They are explicit requirements for crediting relation-internalization-like behavior in controlled toy environments.

The diagnostic question is:

```text
Is the relation structure usable, or is the agent only succeeding under weaker evidence?
```

## Core Criteria

A relation-like structure is credited only when it supports the relevant subset of:

| Criterion | Meaning |
| --- | --- |
| transfer | behavior remains correct under distribution shift |
| counterfactual use | relation variables can be changed and predictions/actions update accordingly |
| editability | local relation edits change target behavior without global collapse |
| auditability | the system can identify relation links used for action |
| temporal indexing | delayed relations include explicit or recoverable time structure |
| uncertainty recognition | the system recognizes when the current relation chain is unverifiable |
| cost-aware inspection | the system chooses informative inspections under budget and cost |

## Gated Scoring

Each stage uses a gated score. If any required structural property fails, the gated score is zero even if ordinary task metrics are high.

This design is intentionally conservative. It prevents a model from receiving a relation-internalization claim based only on high accuracy, plausible explanations, or isolated edit responsiveness.

## Negative Controls

Each experimental group includes baselines that target specific false positives:

| False positive | Negative control style |
| --- | --- |
| prediction success | supervised prediction-only models |
| bottleneck compression | compressed neural models |
| probe readability | shortcut-trained probe-readable models |
| structural memory | memory-like slope/temporal agents |
| generic review text | review-text baselines |
| temporal memory | delayed prediction without editable links |
| fixed-delay template | hard-coded delay logic |
| discovery without uncertainty | relation discovery under partial observability |
| blanket inspection | inspect-any-missing baselines |
| simple inspection heuristic | first-missing, random, risk-first field selection |
| edit-signal responsiveness | edit-pressure model with weak support binding |

## Positive Controls and Their Boundary

Several positive agents are hand-designed relation-structure controls. They are used to define what successful relation-usable behavior looks like under the diagnostic. They are not evidence of spontaneous relation emergence.

The neural stage reduces this limitation by comparing non-handwritten models under prediction, bottleneck, counterfactual, and edit pressures. The current result is conservative: counterfactual training is the strongest neural positive condition, while edit-pressure is mixed.

## Interpretation Rule

The interpretation rule is:

```text
High ordinary performance can motivate further inspection, but only gate-passing structure supports the relation-internalization diagnostic claim.
```

This rule applies equally to successful behavior, readable probes, editable tables, temporal prediction, and inspection policies.
