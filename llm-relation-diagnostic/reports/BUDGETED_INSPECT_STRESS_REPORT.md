# Budgeted Inspect Stress Report

## 1. Purpose

This stage isolates the action-guiding `budgeted_inspect` gate. It tests whether a solver can choose the variable whose inspection makes a relation-chain outcome verifiable under a one-inspection budget.

The stage is a false-positive diagnostic. Passing it would not prove general relation understanding.

## 2. Results

| solver | n | budgeted_inspect | mean_case_score | llm_relation_gated_score |
| --- | --- | --- | --- | --- |
| missingness_template | 15 | 0.200 | 0.200 | 0.000 |
| relation_oracle | 15 | 1.000 | 1.000 | 1.000 |
| surface_audit | 15 | 0.200 | 0.200 | 0.000 |

## 3. Interpretation

- `relation_oracle` is a sanity-check upper bound.
- `missingness_template` should fail when the first missing variable is irrelevant or when the chain is already verifiable.
- `surface_audit` should fail when exact inspect selection and exact audit-link selection are required.

## 4. Boundary

This is a toy black-box diagnostic. It is not evidence of deployment-ready engineering judgment, unrestricted causal discovery, or LLM relation understanding.
