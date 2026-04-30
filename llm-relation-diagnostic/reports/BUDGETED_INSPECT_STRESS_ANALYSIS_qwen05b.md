# Qwen2.5-0.5B Budgeted Inspect Stress Analysis

## 1. Gate Summary

| gate | n | case_score | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| budgeted_inspect | 15 | 0.000 | 0.800 | 0.133 | 0.400 | 0.267 | 1.000 |

## 2. Variant Summary

| variant | n | case_score | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| already_verifiable | 3 | 0.000 | 0.000 | 0.667 | 1.000 | 1.000 | 1.000 |
| many_distractors | 3 | 0.000 | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 |
| mediator_first | 3 | 0.000 | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 |
| noise_first | 3 | 0.000 | 1.000 | 0.000 | 0.333 | 0.333 | 1.000 |
| support_conditioned_noise_first | 3 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## 3. Inspect Error Summary

| inspect_error | n | rate |
| --- | --- | --- |
| wrong_variable | 11 | 0.733 |
| correct | 2 | 0.133 |
| no_inspect | 1 | 0.067 |
| over_inspect | 1 | 0.067 |

## 4. Failed Cases

| solver | seed | case_id | gate | variant | case_score | expected_inspect | actual_inspect | inspect_error | expected_uncertain | actual_uncertain | expected_audit_links | actual_audit_links | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_cpp | 0 | seed0_budgeted_inspect_noise_first | budgeted_inspect | noise_first | 0.0 | q2 | n4 | wrong_variable | True | False | q2 -> n4 | q2 -> n4 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 |  |
| llama_cpp | 0 | seed0_budgeted_inspect_mediator_first | budgeted_inspect | mediator_first | 0.0 | q2 | n4 | wrong_variable | True | False | q2 -> n4 | q2 -> r5 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| llama_cpp | 0 | seed0_budgeted_inspect_many_distractors | budgeted_inspect | many_distractors | 0.0 | q2 | z9 | wrong_variable | True | True | q2 -> n4 | z9=unknown | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 0 | seed0_budgeted_inspect_already_verifiable | budgeted_inspect | already_verifiable | 0.0 | none | none | correct | False | False |  | q2 -> n4 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 0 | seed0_budgeted_inspect_support_conditioned_noise_first | budgeted_inspect | support_conditioned_noise_first | 0.0 | q2 | n4 | wrong_variable | True | False | q2 -> n4 | q2 -> z9 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_noise_first | budgeted_inspect | noise_first | 0.0 | u8 | q2=ta | wrong_variable | True | False | u8 -> q2 | q2 -> z9 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_mediator_first | budgeted_inspect | mediator_first | 0.0 | u8 | q2=ta | wrong_variable | True | True | u8 -> q2 | q2 -> z9 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_many_distractors | budgeted_inspect | many_distractors | 0.0 | u8 | q2=ta | wrong_variable | True | False | u8 -> q2 | q2 -> u8 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_already_verifiable | budgeted_inspect | already_verifiable | 0.0 | none | x7 | over_inspect | False | False |  | q2 -> z9 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 1 | seed1_budgeted_inspect_support_conditioned_noise_first | budgeted_inspect | support_conditioned_noise_first | 0.0 | u8 |  | no_inspect | True | False | u8 -> q2 | none | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_noise_first | budgeted_inspect | noise_first | 0.0 | r5 | p6 | wrong_variable | True | True | r5 -> p6 | none | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_mediator_first | budgeted_inspect | mediator_first | 0.0 | r5 | u8 | wrong_variable | True | False | r5 -> p6 | r5 -> u8 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_many_distractors | budgeted_inspect | many_distractors | 0.0 | r5 | p6 | wrong_variable | True | False | r5 -> p6 | q2 -> z9 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_already_verifiable | budgeted_inspect | already_verifiable | 0.0 | none | none | correct | False | False |  | none | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |  |
| llama_cpp | 2 | seed2_budgeted_inspect_support_conditioned_noise_first | budgeted_inspect | support_conditioned_noise_first | 0.0 | r5 | p6=ta | wrong_variable | True | False | r5 -> p6 | none | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |
