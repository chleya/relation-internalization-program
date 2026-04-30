# Budgeted Inspect Stress Matrix Report

Date: 2026-04-29

## 1. Purpose

This report consolidates the live budgeted-inspect stress results across local
GGUF models. The gate asks whether a model can use a random-symbol relation
chain to choose the single variable that should be inspected under a one-step
inspection budget.

## 2. Local Model Matrix

| model | path | status |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct Q4_K_M | `F:\tmp\models\qwen2.5-0.5b-instruct-q4_k_m.gguf` | completed |
| Qwen2.5-1.5B-Instruct Q4_K_M | `F:\unified-sel-artifacts\gguf_models\qwen2.5-1.5b-instruct-q4_k_m.gguf` | completed |
| Qwen2.5-3B-Instruct Q5_K_M | `F:\unified-sel-artifacts\qwen2.5-3b-instruct-q5_k_m.gguf` | completed |
| Gemma-3-4B-it Q5_K_M | `F:\unified-sel-artifacts\google_gemma-3-4b-it-Q5_K_M (1).gguf` | completed |
| Phi-4-mini-instruct Q4_K_M | `F:\unified-sel-artifacts\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf` | llama.cpp tokenizer regex load failure |

## 3. Score Matrix

| model | passed cases | budgeted_inspect | answer_match | inspect_match | uncertain_match | audit_links_match | gated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B Q4 | 0/15 | 0.000 | 0.800 | 0.133 | 0.400 | 0.267 | 0.000 |
| Qwen2.5-1.5B Q4 | 0/15 | 0.000 | 0.800 | 0.467 | 0.600 | 0.533 | 0.000 |
| Qwen2.5-3B Q5 | 2/15 | 0.133 | 0.800 | 0.133 | 0.933 | 0.400 | 0.000 |
| Gemma-3-4B-it Q5 | 1/15 | 0.067 | 0.867 | 0.333 | 0.867 | 0.333 | 0.000 |

## 4. Failure Pattern

The useful split is component-level:

- Answer fields often look plausible: all completed models score at least 0.800
  on `answer_match`.
- Uncertainty improves with larger models: Qwen3B and Gemma both exceed 0.860
  on `uncertain_match`.
- Inspection selection does not improve monotonically with model size:
  Qwen1.5B has higher `inspect_match` than Qwen3B, but still passes no full
  cases because uncertainty and audit links do not align with the inspection.
- Exact audit links are unstable: models often reverse the link, include extra
  chain fragments, point to the outcome, or point to a distractor relation.

This means the failure is not simply "small models cannot follow JSON." The
models can often answer, can often mark uncertainty, and can sometimes name a
related link. They do not robustly combine those pieces into a correct
cost-constrained inspection policy.

## 5. Output Files

```text
reports/BUDGETED_INSPECT_STRESS_ANALYSIS_qwen05b.md
reports/BUDGETED_INSPECT_STRESS_ANALYSIS_qwen15b.md
reports/BUDGETED_INSPECT_STRESS_ANALYSIS_qwen3b.md
reports/BUDGETED_INSPECT_STRESS_ANALYSIS_gemma3_4b.md

results/budgeted_inspect_stress_failures_qwen05b.csv
results/budgeted_inspect_stress_failures_qwen15b.csv
results/budgeted_inspect_stress_failures_qwen3b.csv
results/budgeted_inspect_stress_failures_gemma3_4b.csv
results/live_matrix_failures.csv
```

## 6. Claim Boundary

This matrix does not prove that the tested model families cannot represent
relations. It shows that the tested local quantized checkpoints do not pass this
toy action-guiding relation gate under the current prompt and scoring protocol.
