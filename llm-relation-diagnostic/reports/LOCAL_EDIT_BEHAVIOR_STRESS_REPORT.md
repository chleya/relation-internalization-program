# Local Edit Behavior Stress Report

## 1. Purpose

This stage isolates the behavior-level `local_edit_locality` gate. It tests whether a solver can apply one support-specific relation edit while preserving other support and unrelated-chain behavior.

The stage is a false-positive diagnostic. Passing it would not prove general relation understanding.

## 2. Results

| solver | n | local_edit_locality | mean_case_score | llm_relation_gated_score |
| --- | --- | --- | --- | --- |
| edit_compliance | 3 | 0.000 | 0.000 | 0.000 |
| edit_no_behavior | 3 | 0.000 | 0.000 | 0.000 |
| global_mapping | 3 | 0.000 | 0.000 | 0.000 |
| relation_oracle | 3 | 1.000 | 1.000 | 1.000 |

## 3. Interpretation

- `relation_oracle` is a sanity-check upper bound.
- `edit_compliance` should fail if it treats edit acknowledgement as global behavior change.
- `edit_no_behavior` should fail if it identifies the edited query but leaves target behavior unchanged.
- A passing solver must return correct post-edit answers for target, other-support, and unrelated queries.

## 4. Boundary

This is a toy black-box diagnostic. It is not evidence of deployment-ready engineering judgment, unrestricted causal discovery, or LLM relation understanding.
