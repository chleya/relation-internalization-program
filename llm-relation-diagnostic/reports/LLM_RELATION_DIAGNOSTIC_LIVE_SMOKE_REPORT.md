# LLM Relation Diagnostic Live Smoke Report

Date: 2026-04-29

## 1. Purpose

This smoke run tested the new black-box LLM relation diagnostic against local
llama.cpp models. The goal was to validate the live model path and inspect early
failure modes, not to make a broad LLM claim.

## 2. Models

```text
Qwen2.5-1.5B-Instruct-Q4_K_M
F:\unified-sel-artifacts\gguf_models\qwen2.5-1.5b-instruct-q4_k_m.gguf

Qwen2.5-3B-Instruct-Q5_K_M
F:\unified-sel-artifacts\qwen2.5-3b-instruct-q5_k_m.gguf
```

Runtime:

```text
F:\AI_Workspace\llama.cpp\build\bin\Release\llama-server.exe
```

## 3. Base Prompt Results

| model | random_symbol_transfer | support_conditioned_binding | counterfactual_use | local_edit_locality | audit_correctness | missing_observation_uncertainty | budgeted_inspect | mean_case_score | gated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B Q4 | 0.667 | 0.667 | 1.000 | 0.000 | 0.000 | 0.333 | 0.000 | 0.381 | 0.000 |
| Qwen2.5-3B Q5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.571 | 0.000 |

## 4. Strict Protocol Smoke Results

Strict protocol prompts add stronger JSON/schema instructions and clearer
variable-selection rules. They are useful for separating formatting/protocol
failure from relation failure. They should not be treated as stronger evidence
than the base prompt results.

| model | random_symbol_transfer | support_conditioned_binding | counterfactual_use | local_edit_locality | audit_correctness | missing_observation_uncertainty | budgeted_inspect | mean_case_score | gated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B Q4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen2.5-1.5B Q4 | 0.333 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.476 | 0.000 |
| Qwen2.5-3B Q5 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 | 0.333 | 0.000 | 0.714 | 0.000 |
| Qwen2.5-3B Q5 strict v2 | 1.000 | 1.000 | 0.667 | 0.333 | 1.000 | 0.333 | 0.000 | 0.619 | 0.000 |
| Gemma-3-4B-it Q5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.714 | 0.000 |

Phi-4-mini Q4 did not produce a relation result because llama.cpp failed while
loading the GGUF tokenizer regex:

```text
Failed to process regex
error loading model vocabulary
```

Output files:

```text
results/llm_relation_summary_qwen15b_smoke.csv
results/llm_relation_records_qwen15b_smoke.csv
results/llm_relation_raw_qwen15b_smoke.jsonl
reports/LLM_RELATION_DIAGNOSTIC_REPORT_qwen15b_smoke.md
reports/LLM_RELATION_DIAGNOSTIC_SELF_AUDIT_qwen15b_smoke.md

results/llm_relation_summary_qwen3b_smoke.csv
results/llm_relation_records_qwen3b_smoke.csv
results/llm_relation_raw_qwen3b_smoke.jsonl
results/llm_relation_summary_qwen3b_smoke.csv
results/llm_relation_summary_strict_qwen05b_strict_smoke.csv
results/llm_relation_summary_strict_qwen15b_strict_smoke.csv
results/llm_relation_summary_strict_qwen3b_strict_smoke.csv
results/llm_relation_summary_strict_qwen3b_strict_v2_smoke.csv
results/llm_relation_summary_strict_gemma3_4b_strict_smoke.csv
```

Note: the first strict local-edit prompt was too explicit about the allowed
query ids. The `qwen3b_strict_v2_smoke` run uses the corrected local-edit
prompt and is the cleaner Qwen3B strict result. The earlier strict table remains
useful for protocol debugging, but local-edit passes from those runs should not
be treated as clean evidence.

## 5. Budgeted Inspect Stress

Because `budgeted_inspect` was the most stable failing gate, a separate stress
case set was added:

```text
configs/budgeted_inspect_stress.yaml
reports/BUDGETED_INSPECT_STRESS_LIVE_REPORT.md
reports/BUDGETED_INSPECT_STRESS_MATRIX_REPORT.md
```

| model | budgeted_inspect | passed cases | gated |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B Q4 | 0.000 | 0/15 | 0.000 |
| Qwen2.5-1.5B Q4 | 0.000 | 0/15 | 0.000 |
| Qwen2.5-3B Q5 | 0.133 | 2/15 | 0.000 |
| Gemma-3-4B-it Q5 | 0.067 | 1/15 | 0.000 |

The stress results show that the live models often recognize uncertainty but do
not reliably choose the relation-chain variable that should be inspected under a
one-inspection budget.

## 6. Local Edit Behavior Stress

The original local-edit smoke gate was strengthened with a behavior-level stress
set:

```text
configs/local_edit_behavior_stress.yaml
reports/LOCAL_EDIT_BEHAVIOR_STRESS_LIVE_REPORT.md
```

| model | local_edit_locality | query_sets_match | answers_by_query_match | gated |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B Q5 | 0.000 | 0.000 | 0.333 | 0.000 |
| Gemma-3-4B-it Q5 | 0.000 | 0.667 | 0.000 | 0.000 |

This separates edit acknowledgement from post-edit behavior. Qwen3B sometimes
computes the right values but marks the wrong queries as affected. Gemma more
often preserves the affected/unchanged shape, but does not produce the correct
post-edit answers.

## 7. Audit Correctness Stress

The audit gate was also strengthened with an exact-link stress set:

```text
configs/audit_correctness_stress.yaml
reports/AUDIT_CORRECTNESS_STRESS_LIVE_REPORT.md
```

| model | audit_correctness | inspect_match | uncertain_match | audit_links_match | gated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B Q4 | 0.000 | 0.400 | 0.533 | 0.200 | 0.000 |
| Qwen2.5-1.5B Q4 | 0.000 | 0.400 | 0.333 | 0.067 | 0.000 |
| Qwen2.5-3B Q5 | 0.000 | 0.333 | 0.333 | 0.200 | 0.000 |
| Gemma-3-4B-it Q5 | 0.000 | 0.400 | 0.467 | 0.333 | 0.000 |

All tested local models fail all full exact-audit cases. They often identify a
related symbol or relation fragment, but fail at least one required part:
inspection target, uncertainty state, exact link direction, support relevance,
or empty audit for irrelevant missing variables.

## 8. Interpretation

The 3B model is materially stronger than the 1.5B model on simple black-box
relation behavior. It passes random-symbol transfer, support-conditioned binding,
counterfactual use, and irrelevant-missing-variable rejection on this smoke set.

Across the tested live runs, no model passes the full gated diagnostic. The most
stable failing gate is:

- budgeted inspect selection under uncertainty.

Audit specificity is also unstable: Qwen3B can pass it under strict protocol, but
Gemma and Qwen1.5B fail it. The base prompts show that local edit locality and
audit behavior are especially sensitive to protocol clarity.

This is useful diagnostic behavior. The harness separates relation answer
generation from editable, auditable, and inspection-guiding relation structure.

## 9. Boundary

This smoke result does not prove that the tested models lack relation
understanding. It shows only that, under the current toy prompts and strict JSON
scoring, they do not pass the full relation-internalization gate.

The next useful step is prompt/protocol hardening, then rerunning the same gates
without changing the thresholds.
