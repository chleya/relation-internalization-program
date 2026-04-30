# Gemma-3-4B Audit Correctness Stress Analysis

## 1. Gate Summary

| gate | n | case_score | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match | answers_by_query_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audit_correctness | 15 | 0.000 | 1.000 | 0.400 | 0.467 | 0.333 | 1.000 | 1.000 |

## 2. Variant Summary

| variant | n | case_score | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match | answers_by_query_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cause_missing | 3 | 0.000 | 1.000 | 0.000 | 0.667 | 1.000 | 1.000 | 1.000 |
| irrelevant_missing | 3 | 0.000 | 1.000 | 0.000 | 1.000 | 0.333 | 1.000 | 1.000 |
| mediator_missing | 3 | 0.000 | 1.000 | 1.000 | 0.667 | 0.000 | 1.000 | 1.000 |
| outcome_observation_missing | 3 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| support_distractor | 3 | 0.000 | 1.000 | 0.000 | 0.000 | 0.333 | 1.000 | 1.000 |

## 3. Inspect Error Summary

| inspect_error | n | rate |
| --- | --- | --- |
| correct | 6 | 0.400 |
| wrong_variable | 6 | 0.400 |
| over_inspect | 3 | 0.200 |

## 4. Failed Cases

| solver | seed | case_id | gate | variant | case_score | expected_inspect | actual_inspect | inspect_error | expected_uncertain | actual_uncertain | expected_audit_links | actual_audit_links | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match | answers_by_query_match | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_cpp | 0 | seed0_audit_mediator_missing | audit_correctness | mediator_missing | 0.0 | q2 | q2 | correct | True | False | q2 -> n4 | p6 -> q2 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 0 | seed0_audit_cause_missing | audit_correctness | cause_missing | 0.0 | p6 | n4 | wrong_variable | True | False | p6 -> q2 | p6 -> q2 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 0 | seed0_audit_support_distractor | audit_correctness | support_distractor | 0.0 | q2 | n4 | wrong_variable | True | False | q2 -> n4 | n4 -> n4 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 0 | seed0_audit_outcome_observation_missing | audit_correctness | outcome_observation_missing | 0.0 | n4 | n4 | correct | True | False | q2 -> n4 |  | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 0 | seed0_audit_irrelevant_missing | audit_correctness | irrelevant_missing | 0.0 | none | n4 | over_inspect | False | False |  | n4 -> sa | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 1 | seed1_audit_mediator_missing | audit_correctness | mediator_missing | 0.0 | u8 | u8 | correct | True | True | u8 -> q2 | r5 -> ru;u8 -> du;q2 -> ta | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 1 | seed1_audit_cause_missing | audit_correctness | cause_missing | 0.0 | r5 | u8 | wrong_variable | True | True | r5 -> u8 | r5 -> u8 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 1 | seed1_audit_support_distractor | audit_correctness | support_distractor | 0.0 | u8 | q2 | wrong_variable | True | False | u8 -> q2 | q2 -> r5 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 1 | seed1_audit_outcome_observation_missing | audit_correctness | outcome_observation_missing | 0.0 | q2 | q2 | correct | True | False | u8 -> q2 | r5 -> ru;u8 -> du | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 1 | seed1_audit_irrelevant_missing | audit_correctness | irrelevant_missing | 0.0 | none | q2 | over_inspect | False | False |  |  | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 2 | seed2_audit_mediator_missing | audit_correctness | mediator_missing | 0.0 | r5 | r5 | correct | True | True | r5 -> p6 | n4 -> p6;r5 -> p6 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 2 | seed2_audit_cause_missing | audit_correctness | cause_missing | 0.0 | n4 | r5 | wrong_variable | True | True | n4 -> r5 | n4 -> r5 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 2 | seed2_audit_support_distractor | audit_correctness | support_distractor | 0.0 | r5 | p6 | wrong_variable | True | False | r5 -> p6 | r5 -> p6 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 2 | seed2_audit_outcome_observation_missing | audit_correctness | outcome_observation_missing | 0.0 | p6 | p6 | correct | True | False | r5 -> p6 |  | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 2 | seed2_audit_irrelevant_missing | audit_correctness | irrelevant_missing | 0.0 | none | p6 | over_inspect | False | False |  | p6 -> p6 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 |  |
