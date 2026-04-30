# Local Edit Behavior Stress Live Report

Date: 2026-04-29

## 1. Purpose

This stress run isolates behavior-level local edit locality. Instead of asking a
model only to list affected query ids, it asks for post-edit answers for three
query ids:

```text
target_chain
other_support
unrelated_chain
```

A passing model must apply the edited support-specific relation to
`target_chain`, preserve the other support, and preserve the unrelated chain.

## 2. Mock Results

| solver | local_edit_locality | gated |
| --- | ---: | ---: |
| relation_oracle | 1.000 | 1.000 |
| edit_compliance | 0.000 | 0.000 |
| edit_no_behavior | 0.000 | 0.000 |
| global_mapping | 0.000 | 0.000 |

The mock false positives cover:

- global edit acknowledgement with wrong affected/unchanged sets;
- correct affected/unchanged sets but unchanged target behavior;
- missing behavior-level answers.

## 3. Live Results

| model | local_edit_locality | query_sets_match | answers_by_query_match | gated |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct-Q5_K_M | 0.000 | 0.000 | 0.333 | 0.000 |
| Gemma-3-4B-it-Q5_K_M | 0.000 | 0.667 | 0.000 | 0.000 |

Output files:

```text
results/local_edit_behavior_stress_summary_qwen3b_local_edit_behavior_stress.csv
results/local_edit_behavior_stress_records_qwen3b_local_edit_behavior_stress.csv
results/local_edit_behavior_stress_raw_qwen3b_local_edit_behavior_stress.jsonl
reports/LOCAL_EDIT_BEHAVIOR_STRESS_ANALYSIS_qwen3b.md

results/local_edit_behavior_stress_summary_gemma3_4b_local_edit_behavior_stress.csv
results/local_edit_behavior_stress_records_gemma3_4b_local_edit_behavior_stress.csv
results/local_edit_behavior_stress_raw_gemma3_4b_local_edit_behavior_stress.jsonl
reports/LOCAL_EDIT_BEHAVIOR_STRESS_ANALYSIS_gemma3_4b.md
```

## 4. Interpretation

Qwen2.5-3B sometimes gives the correct post-edit values, but it marks unrelated
or other-support queries as affected. It does not preserve locality at the query
set level.

Gemma more often gets the affected/unchanged sets right, but its
`answers_by_query` values are wrong. It recognizes the edit locality shape more
often than Qwen3B, but does not reliably execute the edited relation behavior.

This is a stronger local-edit negative than the earlier strict prompt smoke run:
the model must produce behavior after the edit, not only acknowledge that an edit
occurred.

## 5. Boundary

This does not prove that the tested models cannot perform local edits in all
contexts. It shows that, under this random-symbol behavior-level diagnostic,
they do not pass support-specific local edit locality.
