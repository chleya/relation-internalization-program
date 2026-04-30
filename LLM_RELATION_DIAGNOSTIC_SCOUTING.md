# LLM Relation Diagnostic Scouting

Date: 2026-04-29

## Judgment

Local LLM testing is now appropriate, but the first stage should not be framed
as proving that an LLM has relation understanding.

The safer stage question is:

```text
Can an LLM pass relation-internalization gates without relying on language priors,
memorized schemas, self-reported explanations, or prompt-following edit compliance?
```

Current judgment:

- `relation-internalization-program` has clear R1-R3 gates that can be adapted
  to black-box LLM evaluation.
- `unified-sel` already has llama.cpp / LLM calling code and weak-model routing
  evidence, so its runtime layer can be reused.
- LLM testing should be placed inside the false-positive ladder, not treated as
  an immediate positive case.
- The strongest first negative controls are random symbols, support-conditioned
  binding, local edit consistency, missing-observation uncertainty, and
  budget-constrained inspect selection.
- The Qwen2.5 3B local model is the best first primary model. The 1.5B and
  0.5B models are better weak-model negative controls.

## Recommended Next Stage

Recommended stage name:

```text
llm-relation-diagnostic
```

Implementation status:

```text
F:\relation-internalization-program\llm-relation-diagnostic
```

The first harness is implemented. It includes random-symbol cases, a JSON output
contract, scoring metrics, reports, tests, mock false-positive baselines, and an
optional llama.cpp JSON solver.

First baselines:

- `llm_blackbox`
- `llm_fewshot`
- `llm_with_explicit_relation_table`
- `llm_after_edit`

Implemented baseline names in the first runnable harness:

- `relation_oracle`
- `global_mapping`
- `edit_compliance`
- `edit_no_behavior`
- `missingness_template`
- `surface_audit`
- optional `llama_cpp`

First gates:

- random-symbol transfer
- support-conditioned binding
- counterfactual use
- local edit locality
- audit correctness
- missing-observation uncertainty recognition
- budgeted inspect target selection

Claim boundary:

- Do not claim that an LLM has relation-internalized.
- Do not treat explanations, few-shot compliance, edit acknowledgement, or
  chain-of-thought as sufficient evidence.
- Report only whether each LLM passes the toy relation gates and where it fails.

## Local Model Inventory

Primary GGUF artifacts:

```text
F:\unified-sel-artifacts\google_gemma-3-4b-it-Q5_K_M (1).gguf
F:\unified-sel-artifacts\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf
F:\unified-sel-artifacts\qwen2.5-3b-instruct-q5_k_m.gguf
F:\unified-sel-artifacts\gguf_models\qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Additional local model artifacts:

```text
F:\unified-sel\artifacts\qwen2.5-1.5b\model-00001-of-00002.safetensors
F:\unified-sel\artifacts\qwen2.5-1.5b\model-00002-of-00002.safetensors
F:\unified-sel\artifacts\qwen2.5-1.5b\model.safetensors
F:\unified-sel\artifacts\qwen2.5-1.5b\config.json
F:\unified-sel\artifacts\qwen2.5-1.5b\tokenizer.json
F:\unified-sel\artifacts\qwen2.5-1.5b\tokenizer_config.json
F:\unified-sel\artifacts\qwen2.5-1.5b\generation_config.json
F:\unified-sel\artifacts\qwen2.5-1.5b\model.safetensors.index.json
```

Duplicate or sidecar GGUF locations:

```text
F:\unified-sel\double_helix\google_gemma-3-4b-it-Q5_K_M (1).gguf
F:\unified-sel\double_helix\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf
F:\unified-sel\double_helix\ggml-model-i2_s.gguf
F:\tmp\models\qwen2.5-0.5b-instruct-q4_k_m.gguf
F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-fp16.gguf
F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-q8_0.gguf
F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-q6_k.gguf
F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-q5_0.gguf
F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-q4_k_m.gguf
F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-q4_0.gguf
F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-q3_k_m.gguf
F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-q2_k.gguf
```

Non-LLM large model found:

```text
F:\AI_Workspace\SD-WebUI\stable-diffusion-webui\models\Stable-diffusion\v1-5-pruned-emaonly.safetensors
```

Empty or not directly usable as scanned:

```text
F:\unified-sel-artifacts\qwen2.5-1.5b
```

## Runtime Inventory

llama.cpp binaries:

```text
F:\AI_Workspace\llama.cpp\build\bin\Release\llama-server.exe
F:\AI_Workspace\llama.cpp\build\bin\Release\llama-cli.exe
F:\AI_Workspace\BitNet\build\bin\Release\llama-server.exe
```

Current service status when scanned:

```text
127.0.0.1:8081 closed
127.0.0.1:8082 closed
127.0.0.1:8083 closed
```

Existing unified-sel LLM adapter and validation files:

```text
F:\unified-sel\core\llm_solver.py
F:\unified-sel\double_helix\llm_validate.py
F:\unified-sel\experiments\capability\validate_above_filtering.py
F:\unified-sel\experiments\capability\validate_above_filtering_1.5b.py
F:\unified-sel\experiments\capability\validate_above_filtering_3b.py
```

Existing validation results:

```text
F:\unified-sel\results\capability_benchmark\above_filtering_validation.json
F:\unified-sel\results\capability_benchmark\above_filtering_validation_1.5b.json
F:\unified-sel\results\capability_benchmark\above_filtering_validation_3b.json
F:\unified-sel\results\capability_benchmark\llm_routing_1776898846.json
F:\unified-sel\results\capability_benchmark\cross_domain_llm_1776901278.json
```

Observed above/near/below result summary:

```text
Qwen2.5-0.5B-Instruct-Q4_K_M: ABOVE=4, NEAR=0, BELOW=16
Qwen2.5-1.5B-Instruct-Q4_K_M: ABOVE=6, NEAR=0, BELOW=14
Qwen2.5-3B-Instruct-Q5_K_M: ABOVE=5, NEAR=3, BELOW=12
```

First llm-relation-diagnostic mock result:

```text
relation_oracle: llm_relation_gated_score=1.000
global_mapping: llm_relation_gated_score=0.000
edit_compliance: llm_relation_gated_score=0.000
missingness_template: llm_relation_gated_score=0.000
surface_audit: llm_relation_gated_score=0.000
```

First live llama.cpp smoke results:

```text
Qwen2.5-1.5B-Instruct-Q4_K_M:
  random_symbol_transfer=0.667
  support_conditioned_binding=0.667
  counterfactual_use=1.000
  local_edit_locality=0.000
  audit_correctness=0.000
  missing_observation_uncertainty=0.333
  budgeted_inspect=0.000
  mean_case_score=0.381
  llm_relation_gated_score=0.000

Qwen2.5-3B-Instruct-Q5_K_M:
  random_symbol_transfer=1.000
  support_conditioned_binding=1.000
  counterfactual_use=1.000
  local_edit_locality=0.000
  audit_correctness=0.000
  missing_observation_uncertainty=1.000
  budgeted_inspect=0.000
  mean_case_score=0.571
  llm_relation_gated_score=0.000
```

Current interpretation:

- 3B is materially stronger than 1.5B on simple random-symbol and
  support-conditioned behavior.
- Both models still fail the relation-internalization-critical gates:
  local edit locality, exact audit, and budgeted inspect selection.
- This supports using the stage as a false-positive filter before making any
  positive LLM claim.

Strict protocol smoke update:

```text
Qwen2.5-0.5B Q4: mean_case_score=0.000, gated=0.000
Qwen2.5-1.5B Q4: mean_case_score=0.476, gated=0.000
Qwen2.5-3B Q5: mean_case_score=0.714, gated=0.000
Qwen2.5-3B Q5 strict v2: mean_case_score=0.619, gated=0.000
Gemma-3-4B-it Q5: mean_case_score=0.714, gated=0.000
Phi-4-mini Q4: no score; llama.cpp failed to load tokenizer regex
```

The stable hard point across strict live runs is budgeted inspect selection.
Audit specificity remains model- and prompt-sensitive. Strict protocol results
are useful for debugging prompt/schema ambiguity, but base prompt results remain
the safer evidence for relation-internalization claims.

Important correction:

- The first strict local-edit prompt was too explicit about the query-id
  protocol. Local-edit passes from those early strict runs are protocol-assisted
  and should not be treated as clean evidence.
- `qwen3b_strict_v2_smoke` uses the corrected local-edit prompt. Its local edit
  score drops to 0.333 while `budgeted_inspect` remains 0.000.

Budgeted inspect stress update:

```text
Mock:
  relation_oracle: budgeted_inspect=1.000, gated=1.000
  missingness_template: budgeted_inspect=0.200, gated=0.000
  surface_audit: budgeted_inspect=0.200, gated=0.000

Live:
  Qwen2.5-0.5B Q4: budgeted_inspect=0.000, gated=0.000
  Qwen2.5-1.5B Q4: budgeted_inspect=0.000, gated=0.000
  Qwen2.5-3B Q5: budgeted_inspect=0.133, gated=0.000
  Gemma-3-4B-it Q5: budgeted_inspect=0.067, gated=0.000
  Phi-4-mini Q4: no score; llama.cpp tokenizer regex load failure
```

Interpretation:

- The budget stress set catches first-missing-variable templates and generic
  uncertainty/audit templates.
- Qwen2.5-0.5B and Qwen2.5-1.5B pass no full budgeted-inspect stress cases.
- Qwen2.5-3B often knows the task is uncertain, but does not reliably select the
  chain variable to inspect.
- Gemma often emits valid JSON and sometimes identifies an audit link, but its
  inspect target is unstable.
- This strengthens the negative-standard claim: language-level relation answers
  and uncertainty statements are not enough for action-guiding relation
  internalization.

Reusable live tools added:

```text
F:\relation-internalization-program\llm-relation-diagnostic\scripts\run_live_matrix.ps1
F:\relation-internalization-program\llm-relation-diagnostic\src\analyze_results.py
```

Matrix reports:

```text
F:\relation-internalization-program\llm-relation-diagnostic\reports\BUDGETED_INSPECT_STRESS_MATRIX_REPORT.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\BUDGETED_INSPECT_STRESS_ANALYSIS_qwen05b.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\BUDGETED_INSPECT_STRESS_ANALYSIS_qwen15b.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\BUDGETED_INSPECT_STRESS_ANALYSIS_qwen3b.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\BUDGETED_INSPECT_STRESS_ANALYSIS_gemma3_4b.md
```

Local edit behavior stress update:

```text
Mock:
  relation_oracle: local_edit_locality=1.000, gated=1.000
  edit_compliance: local_edit_locality=0.000, gated=0.000
  edit_no_behavior: local_edit_locality=0.000, gated=0.000
  global_mapping: local_edit_locality=0.000, gated=0.000

Live:
  Qwen2.5-3B Q5: local_edit_locality=0.000, query_sets_match=0.000, answers_by_query_match=0.333
  Gemma-3-4B-it Q5: local_edit_locality=0.000, query_sets_match=0.667, answers_by_query_match=0.000
```

Interpretation:

- The earlier strict local-edit smoke result was not enough because it mostly
  tested query-id classification.
- The behavior-level stress requires post-edit answers for target, other-support,
  and unrelated queries.
- Qwen2.5-3B sometimes computes the right post-edit values but does not preserve
  locality in affected/unchanged sets.
- Gemma more often preserves the locality shape, but fails the behavior answers.
- This further supports the distinction between edit acknowledgement and
  editable relation behavior.

Local edit behavior reports:

```text
F:\relation-internalization-program\llm-relation-diagnostic\reports\LOCAL_EDIT_BEHAVIOR_STRESS_LIVE_REPORT.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\LOCAL_EDIT_BEHAVIOR_STRESS_ANALYSIS_qwen3b.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\LOCAL_EDIT_BEHAVIOR_STRESS_ANALYSIS_gemma3_4b.md
```

Audit correctness stress update:

```text
Mock:
  relation_oracle: audit_correctness=1.000, gated=1.000
  surface_audit: audit_correctness=0.000, gated=0.000
  missingness_template: audit_correctness=0.000, gated=0.000
  reverse_audit: audit_correctness=0.200, gated=0.000
  outcome_audit: audit_correctness=0.200, gated=0.000

Live:
  Qwen2.5-0.5B Q4: audit_correctness=0.000, inspect=0.400, uncertain=0.533, audit_links=0.200
  Qwen2.5-1.5B Q4: audit_correctness=0.000, inspect=0.400, uncertain=0.333, audit_links=0.067
  Qwen2.5-3B Q5: audit_correctness=0.000, inspect=0.333, uncertain=0.333, audit_links=0.200
  Gemma-3-4B-it Q5: audit_correctness=0.000, inspect=0.400, uncertain=0.467, audit_links=0.333
```

Interpretation:

- All tested local models fail all full exact-audit stress cases.
- The failure is componentized: models can often name related variables or link
  fragments, but do not jointly satisfy inspect target, uncertainty, exact link
  direction, support relevance, and empty audit on irrelevant missingness.
- This strengthens the distinction between generic uncertainty language and
  relation-specific auditability.

Audit reports:

```text
F:\relation-internalization-program\llm-relation-diagnostic\reports\AUDIT_CORRECTNESS_STRESS_LIVE_REPORT.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\AUDIT_CORRECTNESS_STRESS_ANALYSIS_qwen05b.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\AUDIT_CORRECTNESS_STRESS_ANALYSIS_qwen15b.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\AUDIT_CORRECTNESS_STRESS_ANALYSIS_qwen3b.md
F:\relation-internalization-program\llm-relation-diagnostic\reports\AUDIT_CORRECTNESS_STRESS_ANALYSIS_gemma3_4b.md
```

## Relevant Project Locations

Primary relation project:

```text
F:\relation-internalization-program
```

Most relevant internal stages:

```text
F:\relation-internalization-program\relation-agent-r1
F:\relation-internalization-program\neural-relation-internalization-edit-pressure
F:\relation-internalization-program\neural-relation-probe
F:\relation-internalization-program\relation-internalization-test
F:\relation-internalization-program\temporal-slope-relation-toy
```

Most relevant reports:

```text
F:\relation-internalization-program\relation-agent-r1\reports\R3_ACTIVE_INSPECTION_REPORT.md
F:\relation-internalization-program\relation-agent-r1\reports\R2_1_HARDENING_REPORT.md
F:\relation-internalization-program\neural-relation-internalization-edit-pressure\reports\NEURAL_STAGE_REPORT.md
F:\relation-internalization-program\paper\README_NEURAL_STAGE_UPDATE.md
```

Related reusable projects:

```text
F:\unified-sel
F:\probe_to_boundary_gnn
F:\symbologenesis-boundary
F:\cognitive-execution-engine
F:\cep-cc
F:\sel-lab
```

Immediate reuse judgment:

- `unified-sel`: reuse LLM adapter, llama.cpp server conventions, and
  weak-model diagnostics.
- `relation-agent-r1`: reuse gates and task structure.
- `neural-relation-internalization-edit-pressure`: reuse the false-positive
  boundary around edit responsiveness versus support-conditioned binding.
- `probe_to_boundary_gnn`: useful later for probe methodology, not needed for
  the first LLM black-box diagnostic.
- `cognitive-execution-engine`: useful later for audit/commit governance, not
  needed for the first LLM diagnostic.
- `symbologenesis-boundary`, `cep-cc`, and `sel-lab`: conceptually adjacent,
  but should not be imported as evidence for this stage.
