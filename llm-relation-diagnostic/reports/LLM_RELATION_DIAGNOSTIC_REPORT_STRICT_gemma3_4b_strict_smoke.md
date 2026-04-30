# LLM Relation Diagnostic Report

## 1. Purpose

This stage evaluates black-box LLM outputs against relation-internalization gates using random symbols, support-conditioned rules, edits, audits, and inspection constraints.

The stage is a false-positive diagnostic. Passing it would not prove general relation understanding.

## 2. Results

| solver | n | random_symbol_transfer | support_conditioned_binding | counterfactual_use | local_edit_locality | audit_correctness | missing_observation_uncertainty | budgeted_inspect | mean_case_score | llm_relation_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_cpp | 21 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.714 | 0.000 |

## 3. Interpretation

- `relation_oracle` is a sanity-check upper bound.
- `global_mapping` should fail support-conditioned binding if it ignores support context.
- `edit_compliance` should fail local edit locality if it treats edit acknowledgement as global behavior change.
- `missingness_template` should fail noncritical missingness or budgeted inspect selection if it inspects any missing variable.
- `surface_audit` should fail exact audit-link scoring if it emits generic uncertainty text.

## 4. Boundary

This is a toy black-box diagnostic. It is not evidence of deployment-ready engineering judgment, unrestricted causal discovery, or LLM relation understanding.
