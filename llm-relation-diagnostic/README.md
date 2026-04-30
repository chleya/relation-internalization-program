# LLM Relation Diagnostic

Status: frozen sidecar negative baseline.

This stage tests black-box LLM behavior against relation-internalization gates.

The purpose is not to prove that an LLM understands relations. The purpose is to
ask whether an LLM can pass the same kind of gates without relying on language
priors, memorized schemas, self-reported explanations, or prompt-following edit
compliance.

Read the freeze memo before extending this stage:

```text
reports/LLM_RELATION_DIAGNOSTIC_FREEZE_MEMO.md
```

The stage should now be changed only for bug fixes, reproducibility fixes,
materially different model comparisons, or future comparisons against a
constructive relation agent. It is not the current mainline.

## Gates

- random-symbol transfer
- support-conditioned binding
- counterfactual use
- local edit locality
- audit correctness
- missing-observation uncertainty recognition
- budgeted inspect target selection

## Run

Default run uses mock baselines and does not call a model:

```powershell
pytest -q
python -m src.run_experiment --config configs/base.yaml
python -m src.run_experiment --config configs/strict.yaml
python -m src.run_experiment --config configs/budgeted_inspect_stress.yaml
python -m src.run_experiment --config configs/local_edit_behavior_stress.yaml
python -m src.run_experiment --config configs/audit_correctness_stress.yaml
```

To call a local llama.cpp server, start `llama-server.exe` separately and run:

```powershell
python -m src.run_experiment --config configs/base.yaml --solvers llama_cpp --base-url http://127.0.0.1:8083 --model qwen2.5-3b-instruct-q5_k_m
```

Use `--label qwen3b_smoke` for live runs to avoid overwriting the default mock
baseline outputs.

Use `configs/strict.yaml` to rerun the same gates with a stricter JSON and query
id protocol. The thresholds are unchanged.

Use `configs/budgeted_inspect_stress.yaml` to isolate the action-guiding inspect
gate with missing-variable order, distractors, already-verifiable chains, and
support-conditioned variants.

Use `configs/local_edit_behavior_stress.yaml` to isolate behavior-level local
edit locality with post-edit answers for target, other-support, and unrelated
queries.

Use `configs/audit_correctness_stress.yaml` to isolate exact audit behavior:
inspection target, uncertainty, link direction, support distractors, and
irrelevant missing variables.

To run the budgeted-inspect stress set across local GGUF models with automatic
llama-server startup/shutdown:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_matrix.ps1 -Config configs\budgeted_inspect_stress.yaml -Models qwen05b,qwen15b,qwen3b,gemma3_4b
```

To generate a failure analysis report from a records/raw pair:

```powershell
python -m src.analyze_results --records results\budgeted_inspect_stress_records_qwen3b_budgeted_stress.csv --raw results\budgeted_inspect_stress_raw_qwen3b_budgeted_stress.jsonl --title "Qwen2.5-3B Budgeted Inspect Stress Analysis" --out-md reports\BUDGETED_INSPECT_STRESS_ANALYSIS_qwen3b.md --out-csv results\budgeted_inspect_stress_failures_qwen3b.csv
```

## Expected Outputs

```text
results/llm_relation_records.csv
results/llm_relation_summary.csv
results/llm_relation_raw.jsonl
reports/LLM_RELATION_DIAGNOSTIC_REPORT.md
reports/LLM_RELATION_DIAGNOSTIC_SELF_AUDIT.md
reports/LLM_RELATION_DIAGNOSTIC_FREEZE_MEMO.md
reports/BUDGETED_INSPECT_STRESS_LIVE_REPORT.md
reports/BUDGETED_INSPECT_STRESS_MATRIX_REPORT.md
reports/LOCAL_EDIT_BEHAVIOR_STRESS_LIVE_REPORT.md
reports/AUDIT_CORRECTNESS_STRESS_LIVE_REPORT.md
```

## Claim Boundary

Supported:

- black-box LLM behavior can be evaluated with relation gates;
- prompt compliance, generic audits, global mappings, and blanket inspection can
  be separated from relation-specific behavior in this toy setup.

Unsupported:

- LLM relation understanding;
- deployment readiness;
- real engineering judgment;
- unrestricted causal discovery.
