# V4 Final Report: Bounded Engineering Review Diagnostic

Date: 2026-04-29

## 1. Stage Purpose

V4 moves the relation-internalization mainline from toy action policies into a
bounded engineering-review workflow.

The question is not:

```text
Can the system approve a real slope-engineering plan?
```

The question is:

```text
Can a review output expose relation chains, action effect points, uncertain
links, verification indicators, takeover conditions, and responsibility
boundaries?
```

## 2. Stage Boundary

V4 remains a toy diagnostic.

It does not support:

```text
real geotechnical correctness
real construction-plan approval
real safety prediction
deployment-ready engineering review AI
unrestricted relation discovery
```

## 3. V4 Base Diagnostic

V4 evaluates five toy slope-style review cases.

The base gates require:

```text
relation_chain_specificity >= 0.8
action_point_mapping >= 0.8
uncertainty_takeover_quality >= 0.8
verification_indicator_quality >= 0.7
responsibility_boundary_quality >= 0.9
unsafe_review_rejection >= 0.9
```

Base result:

| agent | gated_v4_score |
| --- | --- |
| generic_review | 0.000 |
| relation_chain_review | 0.000 |
| structural_memory_review | 0.000 |
| surface_warning_review | 0.000 |
| uncertainty_aware_review | 1.000 |

Interpretation:

```text
Relation-chain review alone is not enough. It must also handle uncertainty,
takeover, unsafe-review rejection, and responsibility boundary.
```

## 4. V4.1 Review Hardening

V4.1 attacks false positives from:

```text
schema-template shortcut
fluent but non-specific engineering prose
case-order memorization
responsibility-boundary boilerplate
unsafe approval phrasing
```

Hardening result:

| agent | hardening_v41_gated_score |
| --- | --- |
| uncertainty_aware_review | 1.000 |
| schema_template_review | 0.000 |
| fluent_nonspecific_review | 0.000 |
| case_order_memory_review | 0.000 |
| boundary_boilerplate_review | 0.000 |
| unsafe_approval_review | 0.000 |

Important negative-control result:

```text
case_order_memory_review:
  base_gated_v4_score = 1.000
  case_order_robustness = 0.440
  hardening_v41_gated_score = 0.000
```

This means a reviewer can pass the base V4 diagnostic by memorizing fixed case
order, but V4.1 catches that false positive.

## 5. V4.2 Adversarial Case Mutation

V4.2 mutates the same toy cases without changing their expected review status.

Mutations:

```text
field order changed
irrelevant variables added
relation names paraphrased
hidden unsafe approval phrase inserted
```

Mutation result:

| agent | mutation_v42_gated_score |
| --- | --- |
| uncertainty_aware_review | 1.000 |
| irrelevant_variable_review | 0.000 |
| hidden_approval_echo_review | 0.000 |
| paraphrase_fragile_review | 0.000 |
| field_order_fragile_review | 0.000 |

During V4.2, one real weakness was exposed:

```text
uncertainty_aware_review initially listed irrelevant distractor variables.
```

Fix:

```text
relation-aware reviewers now filter identified variables to relation-relevant
fields instead of copying every observed field.
```

This is a useful correction because V4.2 specifically tests whether irrelevant
variables can distract the review.

## 6. What V4 Supports

V4 supports this narrow claim:

```text
In bounded toy slope-style review cases, relation-chain review can be evaluated
against generic prose, surface warnings, structural memory, schema templates,
case-order memory, irrelevant variables, paraphrased relation names, and unsafe
approval phrasing.
```

It also supports:

```text
Prediction or plausible review text is not enough.
Review outputs must expose usable relation chains, uncertainty points, takeover
conditions, verification indicators, and human responsibility boundaries.
```

## 7. What V4 Does Not Support

V4 does not support:

```text
real slope safety prediction
real engineering approval
real monitoring-system validity
expert replacement
unrestricted engineering relation discovery
robustness to arbitrary natural-language engineering documents
```

## 8. Failure Risks Still Remaining

V4 still has important weaknesses:

```text
cases are curated and synthetic
relation aliases are predefined
mutation operators are hand-designed
reviewers are deterministic baselines
scoring is rule-based and can be gamed
no expert-labeled review corpus is used
no engineering code or numerical slope model is used
```

The strongest remaining false-positive risk is:

```text
A reviewer could overfit the finite V4 schema and mutation operators while still
failing on broader engineering documents.
```

## 9. Verification Snapshot

Commands:

```bash
pytest -q
python -m src.run_v4_review --config configs/v4_review.yaml
python -m src.run_v41_hardening --config configs/v41_hardening.yaml
python -m src.run_v42_mutation --config configs/v42_mutation.yaml
```

Current verification:

```text
pytest -q: 15 passed
V4: uncertainty_aware_review = 1.000, negative controls = 0.000
V4.1: uncertainty_aware_review = 1.000, hardening negatives = 0.000
V4.2: uncertainty_aware_review = 1.000, mutation negatives = 0.000
```

## 10. Final Stage Judgment

V4 should be frozen as:

```text
bounded engineering-review-case diagnostic
```

Do not interpret it as:

```text
engineering-safe AI
real plan approval
real geotechnical understanding
```

## 11. Recommended Next Step

The next stage should not add more V4 scoring unless a concrete reviewer
objection is identified.

Recommended path:

```text
Freeze V4.
Then decide between:
  V5 governance shell integration for logs/approval/replay, or
  V4.3 multi-reviewer disagreement if review comparison becomes necessary.
```

