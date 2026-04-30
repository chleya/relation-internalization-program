# Qwen2.5-3B Local Edit Behavior Stress Analysis

## 1. Gate Summary

| gate | n | case_score | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match | answers_by_query_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_edit_locality | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.333 |

## 2. Failed Cases

| solver | seed | case_id | gate | variant | case_score | expected_inspect | actual_inspect | inspect_error | expected_uncertain | actual_uncertain | expected_audit_links | actual_audit_links | answer_match | inspect_match | uncertain_match | audit_links_match | query_sets_match | answers_by_query_match | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_cpp | 0 | seed0_local_edit_behavior | local_edit_locality | local_edit_behavior | 0.0 |  |  | correct |  |  |  |  | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |  |
| llama_cpp | 1 | seed1_local_edit_behavior | local_edit_locality | local_edit_behavior | 0.0 |  |  | correct |  |  |  |  | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |  |
| llama_cpp | 2 | seed2_local_edit_behavior | local_edit_locality | local_edit_behavior | 0.0 |  |  | correct |  |  |  |  | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |  |
