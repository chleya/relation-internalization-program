# Gemma-3-4B Budgeted Inspect Stress Analysis

## 1. Gate Summary

| gate | n | case_score | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| budgeted_inspect | 15 | 0.067 | 0.867 | 0.333 | 0.867 | 0.333 | 1.000 |

## 2. Variant Summary

| variant | n | case_score | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| already_verifiable | 3 | 0.333 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| many_distractors | 3 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| mediator_first | 3 | 0.000 | 1.000 | 0.333 | 1.000 | 0.333 | 1.000 |
| noise_first | 3 | 0.000 | 1.000 | 0.333 | 1.000 | 0.333 | 1.000 |
| support_conditioned_noise_first | 3 | 0.000 | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 |

## 3. Inspect Error Summary

| inspect_error | n | rate |
| --- | --- | --- |
| correct | 5 | 0.333 |
| no_inspect | 5 | 0.333 |
| wrong_variable | 5 | 0.333 |

## 4. Failed Cases

| solver | seed | case_id | gate | variant | case_score | expected_inspect | actual_inspect | inspect_error | expected_uncertain | actual_uncertain | expected_audit_links | actual_audit_links | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_cpp | 0 | seed0_budgeted_inspect_noise_first | budgeted_inspect | noise_first | 0.0 | q2 | n4 | wrong_variable | True | True | q2 -> n4 | q2 -> n4 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 0 | seed0_budgeted_inspect_mediator_first | budgeted_inspect | mediator_first | 0.0 | q2 | none | no_inspect | True | True | q2 -> n4 |  | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 0 | seed0_budgeted_inspect_many_distractors | budgeted_inspect | many_distractors | 0.0 | q2 | none | no_inspect | True | True | q2 -> n4 | z9 -> n4 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 0 | seed0_budgeted_inspect_already_verifiable | budgeted_inspect | already_verifiable | 0.0 | none | none | correct | False | False |  |  | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 0 | seed0_budgeted_inspect_support_conditioned_noise_first | budgeted_inspect | support_conditioned_noise_first | 0.0 | q2 | n4 | wrong_variable | True | True | q2 -> n4 | query -> support | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_noise_first | budgeted_inspect | noise_first | 0.0 | u8 | u8 | correct | True | True | u8 -> q2 | q2 -> u8 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_mediator_first | budgeted_inspect | mediator_first | 0.0 | u8 | u8 | correct | True | True | u8 -> q2 | q2 -> u8 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_many_distractors | budgeted_inspect | many_distractors | 0.0 | u8 | none | no_inspect | True | True | u8 -> q2 |  | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_already_verifiable | budgeted_inspect | already_verifiable | 0.0 | none | none | correct | False | False |  |  | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_support_conditioned_noise_first | budgeted_inspect | support_conditioned_noise_first | 0.0 | u8 | q2 | wrong_variable | True | False | u8 -> q2 |  | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_noise_first | budgeted_inspect | noise_first | 0.0 | r5 | none | no_inspect | True | True | r5 -> p6 |  | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_mediator_first | budgeted_inspect | mediator_first | 0.0 | r5 | p6 | wrong_variable | True | True | r5 -> p6 | r5 -> p6 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_many_distractors | budgeted_inspect | many_distractors | 0.0 | r5 | none | no_inspect | True | True | r5 -> p6 | q2 -> z9 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_support_conditioned_noise_first | budgeted_inspect | support_conditioned_noise_first | 0.0 | r5 | p6 | wrong_variable | True | False | r5 -> p6 |  | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
