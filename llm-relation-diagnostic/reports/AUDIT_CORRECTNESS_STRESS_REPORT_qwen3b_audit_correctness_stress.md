# Audit Correctness Stress Report

## 1. Purpose

This stage isolates the `audit_correctness` gate. It tests whether a solver can identify the exact unverifiable relation link, the right inspection variable, and the absence of an audit link when the missing variable is irrelevant.

The stage is a false-positive diagnostic. Passing it would not prove general relation understanding.

## 2. Results

| solver | n | audit_correctness | mean_case_score | llm_relation_gated_score |
| --- | --- | --- | --- | --- |
| llama_cpp | 15 | 0.000 | 0.000 | 0.000 |

## 3. Interpretation

- `relation_oracle` is a sanity-check upper bound.
- `surface_audit` should fail if it emits generic uncertainty text.
- `missingness_template` should fail if it audits the first missing variable without checking relation relevance.
- `reverse_audit` should fail if it names a related link in the wrong direction.
- `outcome_audit` should fail if it defaults to inspecting the outcome instead of the blocking chain variable.

## 4. Boundary

This is a toy black-box diagnostic. It is not evidence of deployment-ready engineering judgment, unrestricted causal discovery, or LLM relation understanding.
