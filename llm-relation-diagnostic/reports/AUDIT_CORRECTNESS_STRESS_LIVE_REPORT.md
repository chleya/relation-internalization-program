# Audit Correctness Stress Live Report

Date: 2026-04-29

## 1. Purpose

This stress run isolates exact relation audit behavior. It tests whether a
solver can:

- choose the inspection variable that blocks verification;
- mark uncertainty only when the relation chain cannot be verified;
- name the exact unverifiable relation link in the correct direction;
- return no audit link when the missing variable is irrelevant.

## 2. Case Set

Config:

```text
configs/audit_correctness_stress.yaml
```

Variants per seed:

```text
mediator_missing
cause_missing
support_distractor
outcome_observation_missing
irrelevant_missing
```

Seeds:

```text
0, 1, 2
```

Total cases per solver: 15.

## 3. Mock Baseline Results

| solver | audit_correctness | passed cases | gated |
| --- | ---: | ---: | ---: |
| relation_oracle | 1.000 | 15/15 | 1.000 |
| surface_audit | 0.000 | 0/15 | 0.000 |
| missingness_template | 0.000 | 0/15 | 0.000 |
| reverse_audit | 0.200 | 3/15 | 0.000 |
| outcome_audit | 0.200 | 3/15 | 0.000 |

The 0.200 scores are from `irrelevant_missing` cases where no audit link is
required; those baselines still fail the gated stress set.

## 4. Live Model Matrix

| model | passed cases | audit_correctness | inspect_match | uncertain_match | audit_links_match | gated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B Q4 | 0/15 | 0.000 | 0.400 | 0.533 | 0.200 | 0.000 |
| Qwen2.5-1.5B Q4 | 0/15 | 0.000 | 0.400 | 0.333 | 0.067 | 0.000 |
| Qwen2.5-3B Q5 | 0/15 | 0.000 | 0.333 | 0.333 | 0.200 | 0.000 |
| Gemma-3-4B-it Q5 | 0/15 | 0.000 | 0.400 | 0.467 | 0.333 | 0.000 |

Phi-4-mini Q4 remains unrunnable in this setup because llama.cpp fails while
loading the tokenizer regex.

## 5. Failure Pattern

The models often produce a related symbol or a plausible relation fragment, but
the full audit contract fails. Typical failures:

- choosing the outcome instead of the missing chain variable;
- naming the right link in a chain fragment with extra links or values;
- reversing the direction of the relation link;
- auditing distractor-support links;
- auditing irrelevant missing variables;
- omitting `uncertain` or setting it inconsistently with the audit.

Qwen3B is relatively better at uncertainty on mediator-missing cases, but weak
on support distractors and irrelevant missing variables. Gemma names exact links
more often on some cause-missing and support-distractor cases, but still fails
because inspect target or uncertainty is wrong.

## 6. Output Files

```text
reports/AUDIT_CORRECTNESS_STRESS_ANALYSIS_qwen05b.md
reports/AUDIT_CORRECTNESS_STRESS_ANALYSIS_qwen15b.md
reports/AUDIT_CORRECTNESS_STRESS_ANALYSIS_qwen3b.md
reports/AUDIT_CORRECTNESS_STRESS_ANALYSIS_gemma3_4b.md

results/audit_correctness_stress_failures_qwen05b.csv
results/audit_correctness_stress_failures_qwen15b.csv
results/audit_correctness_stress_failures_qwen3b.csv
results/audit_correctness_stress_failures_gemma3_4b.csv
```

## 7. Boundary

This does not prove that the tested models cannot audit relations in all
contexts. It shows that these local quantized checkpoints do not pass this
random-symbol exact-audit gate under the current prompt and scoring protocol.
