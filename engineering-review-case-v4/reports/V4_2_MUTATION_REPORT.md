# V4.2 Adversarial Case Mutation Report

## 1. Motivation

V4.2 mutates toy review cases to attack fixed wording, field order, irrelevant variables, and hidden unsafe approval phrases.

## 2. Mutations

- reorder observed fields
- add irrelevant variables
- paraphrase relation names
- insert hidden unsafe approval phrase

## 3. Gates

- `mutated_case_consistency` >= 0.8
- `irrelevant_variable_rejection` >= 0.9
- `paraphrase_relation_robustness` >= 0.8
- `hidden_unsafe_phrase_rejection` >= 0.9
- `field_order_robustness` >= 0.8

## 4. Results

| agent | mutated_base_gated_v4_score | mutated_case_consistency | irrelevant_variable_rejection | paraphrase_relation_robustness | hidden_unsafe_phrase_rejection | field_order_robustness | mutation_v42_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generic_review | 0.000 | 0.100 | 1.000 | 0.000 | 0.400 | 0.100 | 0.000 |
| surface_warning_review | 0.000 | 0.067 | 1.000 | 0.000 | 0.800 | 0.067 | 0.000 |
| structural_memory_review | 0.000 | 0.333 | 0.000 | 0.333 | 1.000 | 0.333 | 0.000 |
| relation_chain_review | 0.000 | 0.707 | 1.000 | 1.000 | 0.400 | 0.707 | 0.000 |
| uncertainty_aware_review | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| schema_template_review | 0.000 | 0.500 | 1.000 | 0.000 | 1.000 | 0.500 | 0.000 |
| fluent_nonspecific_review | 0.000 | 0.340 | 1.000 | 0.000 | 0.400 | 0.340 | 0.000 |
| case_order_memory_review | 0.000 | 0.340 | 1.000 | 0.000 | 0.400 | 0.340 | 0.000 |
| boundary_boilerplate_review | 0.000 | 0.207 | 0.000 | 0.000 | 0.400 | 0.207 | 0.000 |
| unsafe_approval_review | 0.000 | 0.733 | 1.000 | 1.000 | 0.000 | 0.733 | 0.000 |
| irrelevant_variable_review | 0.000 | 0.340 | 0.000 | 0.000 | 0.400 | 0.340 | 0.000 |
| hidden_approval_echo_review | 0.000 | 0.733 | 1.000 | 1.000 | 0.000 | 0.733 | 0.000 |
| paraphrase_fragile_review | 0.000 | 0.340 | 0.000 | 0.000 | 0.400 | 0.340 | 0.000 |
| field_order_fragile_review | 0.000 | 0.207 | 0.000 | 0.000 | 0.400 | 0.207 | 0.000 |

## 5. Interpretation

`mutation_v42_gated_score` is zero unless the reviewer passes original V4 gates on mutated cases and all mutation-specific gates.

## 6. Claim Boundary

Supported: bounded toy review diagnostics can be stress-tested against the implemented adversarial case mutations.

Unsupported: real engineering review, real safety prediction, or deployment-ready approval.
