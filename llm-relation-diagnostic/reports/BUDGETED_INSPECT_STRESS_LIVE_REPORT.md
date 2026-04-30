# Budgeted Inspect Stress Live Report

Date: 2026-04-29

## 1. Purpose

This stress run isolates the `budgeted_inspect` gate because it was the most
stable live-model failure in the first LLM relation diagnostic smoke tests.

The stress set varies missing-variable order, irrelevant distractors,
already-verifiable chains, and support-conditioned chains. The intended
behavior is not generic uncertainty. The model must choose the chain variable
whose inspection makes the target outcome verifiable, or choose `none` when the
chain is already verifiable.

## 2. Case Set

Config:

```text
configs/budgeted_inspect_stress.yaml
```

Variants per seed:

```text
noise_first
mediator_first
many_distractors
already_verifiable
support_conditioned_noise_first
```

Seeds:

```text
0, 1, 2
```

Total cases per solver: 15.

## 3. Mock Baseline Results

| solver | budgeted_inspect | passed cases | gated |
| --- | ---: | ---: | ---: |
| relation_oracle | 1.000 | 15/15 | 1.000 |
| missingness_template | 0.200 | 3/15 | 0.000 |
| surface_audit | 0.200 | 3/15 | 0.000 |

Interpretation:

- `missingness_template` only passes the `mediator_first` cases by selecting the
  first missing variable.
- `surface_audit` only passes the `already_verifiable` cases where no audit link
  is required.
- The stress set catches both "inspect any missing variable" and "say generic
  uncertainty" false positives.

## 4. Live Model Results

| model | budgeted_inspect | passed cases | gated |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct-Q4_K_M | 0.000 | 0/15 | 0.000 |
| Qwen2.5-1.5B-Instruct-Q4_K_M | 0.000 | 0/15 | 0.000 |
| Qwen2.5-3B-Instruct-Q5_K_M | 0.133 | 2/15 | 0.000 |
| Gemma-3-4B-it-Q5_K_M | 0.067 | 1/15 | 0.000 |

Phi-4-mini Q4 did not run because llama.cpp failed to load the tokenizer regex.

Output files:

```text
results/budgeted_inspect_stress_summary_qwen05b_budgeted_stress.csv
results/budgeted_inspect_stress_records_qwen05b_budgeted_stress.csv
results/budgeted_inspect_stress_raw_qwen05b_budgeted_stress.jsonl

results/budgeted_inspect_stress_summary_qwen15b_budgeted_stress.csv
results/budgeted_inspect_stress_records_qwen15b_budgeted_stress.csv
results/budgeted_inspect_stress_raw_qwen15b_budgeted_stress.jsonl

results/budgeted_inspect_stress_summary_qwen3b_budgeted_stress.csv
results/budgeted_inspect_stress_records_qwen3b_budgeted_stress.csv
results/budgeted_inspect_stress_raw_qwen3b_budgeted_stress.jsonl

results/budgeted_inspect_stress_summary_gemma3_4b_budgeted_stress.csv
results/budgeted_inspect_stress_records_gemma3_4b_budgeted_stress.csv
results/budgeted_inspect_stress_raw_gemma3_4b_budgeted_stress.jsonl

results/live_matrix_failures.csv
```

See also:

```text
reports/BUDGETED_INSPECT_STRESS_MATRIX_REPORT.md
reports/BUDGETED_INSPECT_STRESS_ANALYSIS_qwen05b.md
reports/BUDGETED_INSPECT_STRESS_ANALYSIS_qwen15b.md
reports/BUDGETED_INSPECT_STRESS_ANALYSIS_qwen3b.md
reports/BUDGETED_INSPECT_STRESS_ANALYSIS_gemma3_4b.md
```

## 5. Failure Pattern

Qwen2.5-0.5B produces plausible answer fields in many cases, but mostly selects
wrong variables and has weak uncertainty recognition.

Qwen2.5-1.5B has the highest raw `inspect_match` among the Qwen stress runs, but
still passes no full cases because uncertainty and audit links do not align with
the selected inspect target.

Qwen2.5-3B usually recognizes uncertainty, but the selected `inspect` target is
unstable. It often returns `none`, the outcome variable, or an unrelated
variable. It also inspects a mediator in already-verifiable cases where no
inspection is needed.

Gemma often emits valid JSON and sometimes names the right audit link, but its
inspection target is also unstable. It tends to select the outcome, `none`, or a
symbol from the wrong link. It passed only one already-verifiable case.

The shared failure is not just formatting. Both models can comply with the JSON
contract, but do not robustly convert a relation chain into a cost-constrained
inspection policy.

## 6. Boundary

This does not prove that either model lacks relation understanding in general.
It shows that, under this toy random-symbol diagnostic, the tested local models
do not satisfy the action-guiding inspect-selection requirement.
