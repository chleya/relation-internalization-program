# LLM Relation Diagnostic Freeze Memo

Date: 2026-04-29

## Decision

Freeze `llm-relation-diagnostic` as a sidecar negative baseline.

Do not continue expanding black-box prompt stress tests as the main research
route.

## Why

The stage has served its useful purpose:

- local GGUF models can be run through the relation gates;
- mock false-positive baselines are separated from the oracle;
- live models fail the full gates under strict scoring;
- failures are behavioral, not just JSON-format failures;
- failure modes are recorded for budgeted inspect, exact audit, and local edit
  behavior.

Continuing to add more prompt variants would mostly show:

```text
more toy prompts -> more black-box LLM failures
```

That is a valid negative result, but it does not advance the constructive
question:

```text
Can a system learn and expose usable internal relations?
```

## Current Evidence Summary

Core smoke:

```text
Qwen2.5-3B Q5: simple transfer/support/counterfactual behavior is possible,
but full gated relation internalization fails.
```

Budgeted inspect stress:

```text
Qwen2.5-0.5B Q4: 0/15
Qwen2.5-1.5B Q4: 0/15
Qwen2.5-3B Q5: 2/15
Gemma-3-4B-it Q5: 1/15
```

Local edit behavior stress:

```text
Qwen2.5-3B Q5: 0/3
Gemma-3-4B-it Q5: 0/3
```

Audit correctness stress:

```text
Qwen2.5-0.5B Q4: 0/15
Qwen2.5-1.5B Q4: 0/15
Qwen2.5-3B Q5: 0/15
Gemma-3-4B-it Q5: 0/15
```

Phi-4-mini Q4 did not run because llama.cpp failed while loading the tokenizer
regex.

## What The Stage Supports

Supported:

- black-box LLM prompts can be tested against relation-internalization gates;
- answer correctness, uncertainty language, audit text, edit acknowledgement,
  and action-guiding inspect behavior can be separated;
- the tested local models do not pass the full toy gates;
- the diagnostic is useful as a negative baseline for future constructive
  systems.

Unsupported:

- proof that LLMs cannot internalize relations;
- proof that larger hosted models would fail;
- proof of any positive relation-internalization mechanism;
- deployment or engineering competence claims.

## Frozen Scope

Keep:

```text
configs/base.yaml
configs/strict.yaml
configs/budgeted_inspect_stress.yaml
configs/local_edit_behavior_stress.yaml
configs/audit_correctness_stress.yaml
scripts/run_live_matrix.ps1
src/analyze_results.py
reports/*LIVE_REPORT.md
reports/*ANALYSIS*.md
```

Only change this sidecar for:

- bug fixes;
- reproducibility fixes;
- adding a materially different model family if needed for comparison;
- comparing a future constructive agent against the same gates.

Do not add more black-box prompt stress sets unless there is a specific new
false-positive theory that the current reports do not already cover.

## Next Mainline

Return to the constructive agent route:

```text
R2.1 partial-observability hardening
```

Then move toward a neural constructive route only after the symbolic/explicit
agent line has a harder uncertainty and inspect baseline:

```text
neural relation agent with counterfactual, edit, audit, and inspect objectives
```
