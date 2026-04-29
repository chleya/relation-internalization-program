# When Prediction Is Not Relation Internalization:
# Editable and Auditable Diagnostics for Relational Agents

# Abstract

When should an agent be credited with internalizing a relation rather than merely predicting outcomes, memorizing contexts, following surface cues, or producing plausible review text? We study this question through a staged chain of controlled toy diagnostics. Each stage introduces false-positive baselines that can look strong under ordinary metrics but fail relation-internalization gates.

The diagnostic chain covers static explicit relation tables, neural hidden-state probes, engineering-style slope relation chains, delayed temporal relations, partial observability, relation-specific uncertainty, and active inspection under cost. Across these settings, task success, prediction accuracy, probe readability, memory, surface shortcuts, plausible review text, temporal prediction, blanket inspection, and simple inspection heuristics are insufficient evidence of relation internalization. The supported claim is narrow: in controlled toy diagnostic environments, relation internalization can be operationalized as relation structures that are transferable, counterfactual, editable, auditable, temporally indexed, uncertainty-aware, and cost-aware.

# 1. Introduction

Ordinary task accuracy is too weak as evidence for relation internalization. A policy can achieve high success by memorizing contexts, exploiting a correlated surface cue, fitting a shortcut, or producing plausible explanatory text after the fact. A neural hidden state can contain linearly readable relation information without using that information in a behaviorally specific way. A temporal agent can predict delayed outcomes without having editable delayed links. A partial-observability agent can discover relations and still automate unsafely when the current relation chain is unverifiable.

This report treats relation internalization as an operational diagnostic standard, not as a full theory of intelligence. The question is not whether an agent "understands" relations in a broad sense. The question is whether a candidate internal structure is usable in the ways relation claims require: transfer, counterfactual use, intervention, edit, audit, action guidance, temporal indexing, uncertainty handling, and cost-aware inspection.

Every stage was designed around a baseline that should pass if ordinary performance were enough, but fails once relation internalization is required.

The result is a paper-style diagnostic methodology. It does not claim real slope monitoring, real geotechnical safety capability, deployment-ready engineering AI, general causal discovery, object permanence, LLM replacement, or proof that large neural systems naturally internalize relations.

# 2. Operational Definition

We define relation internalization as:

```text
External relations become internal usable structures.
```

Usable means that the relation structure supports:

- transfer;
- counterfactual use;
- intervention/edit;
- audit;
- action guidance;
- temporal indexing when relations are delayed;
- uncertainty handling under partial observability;
- cost-aware inspection under budget.

This definition intentionally distinguishes relation internalization from:

- high prediction accuracy;
- probe readability;
- context memory;
- surface cue use;
- generic review language;
- blanket inspection.

A model or agent can pass an ordinary task metric while failing this diagnostic. The diagnostic therefore uses gated scores: if a required structural property fails, the relation-internalization score is zero even when other metrics are high.

# 3. Diagnostic Ladder

## Experiment 1: Static Relation Internalization

This experiment includes the food-world explicit relation table and the R1 / R1.1 / R1.2 active relation agent line.

The food-world setting tests whether a table over food-resource relations can transfer out of distribution, resist spurious cues, support counterfactual resource inference, support edits, and expose relation-table alignment. The strongest explicit relation-table results come from `relation` and `robust_wide_relation`; memory, fitting, predictive, and decision-tree baselines fail gated internalization despite ordinary success on parts of the task.

The R1 line then moves to an active non-LLM agent. R1 requires action success, OOD action success, relation recovery, counterfactual accuracy, edit success, and active exploration. R1.1 adds hidden confounder rejection, rule reversal, intervention cost tradeoffs, candidate expansion precision, and active discovery. R1.2 removes hand-supplied true-link dependence by requiring relation candidates to be discovered from observed transitions.

Purpose:

- Rule out memory.
- Rule out fitting and prediction-only behavior.
- Rule out shortcut policies.
- Rule out no-exploration relation tables.
- Rule out dependence on predefined `TRUE_LINK` candidates.

## Experiment 2: Neural Readability vs Causal Use

This experiment includes the neural hidden-state relation probe, causal subspace intervention, neural-to-table extraction, and the newer counterfactual/edit-pressure neural stage.

The neural probe asks whether a small classifier encodes the true `texture/wet -> resource` relation. It then tests whether the relation subspace is behaviorally relevant by removing relation and nuisance subspaces. The extraction bridge asks whether hidden-state relation information can be converted into an explicit editable table.

The counterfactual/edit-pressure stage addresses a weakness of the earlier hand-written agents: positive evidence should not come only from systems where relation tables and edit interfaces were directly built in. In the current toy result, `counterfactual_training` is the strongest non-handwritten positive result, while `edit_pressure_training` is mixed: it supports transfer, table extraction, table edit, and edit locality, but relation-subspace intervention is unstable across seeds.

Purpose:

- Rule out probe readability alone.
- Rule out shortcut-readable but behaviorally non-specific relation representations.
- Rule out extracted tables that are editable but wrong.
- Rule out the assumption that extracted-table editability automatically implies a stable causal linear relation subspace.

## Experiment 3: Engineering-Style Relation Chains

This experiment includes the slope-relation-toy agents:

- `relation_chain`;
- `learned_links`;
- `structural_memory`;
- `generic_review`;
- `surface`;
- other simple baselines.

The slope toy uses engineering-style variables and actions, but remains a toy diagnostic. It tests relation-chain action, OOD transfer, spurious attack robustness, counterfactuals, edits, relation audit, irrelevant-link rejection, noisy observations, and review consistency.

Purpose:

- Rule out generic review language.
- Rule out structural memory.
- Rule out surface labels.
- Rule out plausible text as sufficient evidence of relation internalization.

## Experiment 4: Temporal Relation Internalization

This experiment includes temporal-slope-relation-toy V2 and V2.1 Reviewer Hardening.

V2 tests delayed relation chains using temporal OOD success, surface shortcut rejection, delayed counterfactual accuracy, delay edit success, and temporal audit. V2.1 hardens the result with variable delays, false delay shortcut rejection, multi-link delay edits, temporal audit consistency, and anti-template generalization.

Purpose:

- Rule out same-step logic.
- Rule out fixed-delay templates.
- Rule out temporal memory without editable/auditable delay links.
- Rule out false temporal shortcuts.
- Rule out audit strings without real time indexes.

## Experiment 5: Uncertainty and Active Inspection

This experiment includes R2 partial observability, R2.1 relation-specific uncertainty versus blanket inspection, and R3 active inspection selection under cost.

R2 tests whether discovered relations remain usable when observations are missing, noisy, or conflicting. R2.1 tests whether inspection is relation-specific rather than a blanket response to any unknown field. R3 adds multi-field missingness, inspection target selection, sequential inspection, a rule-based inspection value function, limited budget, inspection cost, overinspection penalty, and unsafe automation penalty.

Purpose:

- Rule out relation discovery without uncertainty handling.
- Rule out blanket inspection.
- Rule out first-missing, random-field, and risk-first inspection heuristics.
- Rule out unsafe automation under unverifiable relation chains.

# 4. Results Summary

This section uses only existing result CSVs and reports.

## Neural probe

Existing reports show:

- Base mode OOD accuracy: `1.000`.
- Base mode spurious attack accuracy: `1.000`.
- Base mode relation-subspace drop: about `0.483`.
- Base mode gated neural relation score: about `0.819`.
- Shortcut mode gated score: `0.000`.
- Extracted table base gated extraction: `1.000`.
- Extracted table shortcut gated extraction: `0.000`.

Interpretation:
The shortcut model remains probe-readable, but the gate rejects it because the relation representation is not behaviorally specific under transfer and intervention tests.

## Neural counterfactual/edit-pressure stage

Existing reports show:

- `pure_prediction`: gated score `0.000`; train accuracy `0.996`, OOD `0.913`, shortcut rejection `0.636`, relation-subspace drop `0.087`.
- `prediction_bottleneck`: gated score `0.000`; OOD/shortcut/reversal all `1.000`, but relation and nuisance subspace drops are both `0.507`.
- `counterfactual_training`: gated score `0.981`; OOD, shortcut, reversal, counterfactual, table, edit, and locality metrics all `1.000`; relation-subspace drop `0.406`; nuisance-subspace drop `0.000`.
- `edit_pressure_training`: gated score `0.200`; OOD, shortcut, reversal, table, edit, and locality metrics all `1.000`; relation-subspace drop `0.235`; nuisance-subspace drop `0.000`.
- V1.1 failure localization shows `edit_pressure_training` fails the relation-subspace gate in 4 of 5 seeds.

Interpretation:
Counterfactual training is the strongest current non-handwritten neural positive result. Edit-pressure training is mixed: it produces editable extracted-table behavior, but does not reliably produce a stable causal linear relation subspace. This adds a new false-positive category: editable extracted tables are not automatically stable causal subspaces.

## Slope toy

Existing reports show:

- `relation_chain` / `learned_links` gated slope score: about `0.98`.
- `structural_memory` has high OOD/spurious performance but gated score `0.000`.
- `generic_review` gated score: `0.000`.

Interpretation:
The slope toy separates relation-chain use from structural memory and plausible review language.

## Temporal V2/V2.1

Existing reports show:

- `delayed_relation_chain` and `learned_delayed_links` gated temporal / hardening scores: `1.000`.
- `structural_memory_temporal` has high temporal prediction but hardening score `0.000`.
- `instant_relation_chain` hardening score: `0.000`.

Interpretation:
Temporal prediction is not temporal relation internalization. Editable and auditable delay links are required.

## R1-R3

Existing reports show:

- R1 `relation_agent` gated score: `0.972`.
- R1.1 `relation_agent` hardening score: `1.000`.
- R1.2 `discovery_relation_agent` discovery score: `1.000`.
- R2 `discovery_relation_agent` partial observation success: `0.967`, but gated score: `0.000`.
- R2 `uncertainty_discovery_agent` gated score: `0.982`.
- R2.1 `relation_specific_uncertainty_agent` gated score: `0.933`.
- R2.1 `missing_always_inspect` gated score: `0.000`.
- R3 `active_inspection_agent` gated score: `0.933`.
- R3 `random_inspect_field`, `first_missing_inspect`, `missing_always_inspect`, and `risk_first_inspect` baselines: gated score `0.000`.

Interpretation:
The R1-R3 line separates active relation learning, relation discovery, uncertainty recognition, relation-specific inspection, and budgeted inspection selection.

# 5. What Each Stage Rules Out

| Stage | Positive agent/result | Negative control | Ordinary metric that could look good | Gate that fails | Alternative explanation ruled out |
| --- | --- | --- | --- | --- | --- |
| Neural probe | Base neural model, gated score about `0.819`; extracted table gated `1.000` | Neural shortcut | Probe readability remains high | Gated neural relation score `0.000`; extraction gate `0.000` | Readable relation information alone is not behaviorally specific relation internalization. |
| Neural counterfactual/edit-pressure | `counterfactual_training`, gated `0.981` | `pure_prediction` | Train/OOD accuracy can be high | Shortcut and relation-subspace gates | Prediction alone is not relation internalization. |
| Neural counterfactual/edit-pressure | `counterfactual_training`, gated `0.981` | `prediction_bottleneck` | OOD, shortcut, reversal, table, edit metrics can be high | Nuisance-subspace gate | Compression can entangle relation and nuisance factors. |
| Neural counterfactual/edit-pressure | `counterfactual_training`, gated `0.981` | `edit_pressure_training` | Transfer, table extraction, table edit, and locality all look good | Relation-subspace gate unstable across seeds | Editable extracted tables are not automatically stable causal linear subspaces. |
| Food-world static | `relation` / `robust_wide_relation` pass gated internalization | Memory, fitting, predictive, decision tree | Base task success and some OOD success | Edit, counterfactual, relation alignment, shuffle-drop gates | Prediction or context memory is not editable relation structure. |
| Slope toy | `relation_chain` / `learned_links`, gated about `0.98` | `structural_memory` | OOD and spurious performance can be high | Edit/audit/review gates, gated `0.000` | Structural memory is not auditable/editable relation internalization. |
| Slope toy | `relation_chain` / `learned_links` | `generic_review` | Plausible review score language | Relation behavior and consistency gates, gated `0.000` | Review-like text is not relation use. |
| Temporal V2/V2.1 | `delayed_relation_chain`, `learned_delayed_links`, hardening `1.000` | `structural_memory_temporal` | High temporal prediction and generalization | Delay edit and temporal audit gates, hardening `0.000` | Temporal memory is not editable delayed relation structure. |
| Temporal V2/V2.1 | `delayed_relation_chain`, `learned_delayed_links` | `instant_relation_chain` | Some relation-chain behavior | Variable delay, false shortcut, edit, audit gates | Same-step relation logic cannot substitute for delayed relations. |
| R1 | `relation_agent`, gated `0.972` | Shortcut/passive/random | Action success can be high for shortcut/passive cases | Relation recovery, counterfactual, edit, exploration gates | Task success is not active relation internalization. |
| R1.1 | `relation_agent`, hardening `1.000` | `relation_no_explore` | Candidate precision can be high | Active discovery gate | A relation table without exploration is not enough. |
| R1.2 | `discovery_relation_agent`, discovery `1.000` | Original `relation_agent` in discovery setting | Action success remains high | New-link discovery and adaptive exploration gates | Hand-supplied relation candidates are not discovery. |
| R2 | `uncertainty_discovery_agent`, gated `0.982` | `discovery_relation_agent` | Partial observation success `0.967` | Inspection recall, unsafe automation, uncertainty audit gates | Relation discovery alone is not safe under partial observability. |
| R2.1 | `relation_specific_uncertainty_agent`, gated `0.933` | `missing_always_inspect` | Critical-missing inspection recall can be high | Noncritical no-inspect, inspection precision, cost gates | Blanket inspection is not relation-specific uncertainty. |
| R3 | `active_inspection_agent`, gated `0.933` | `first_missing_inspect`, `random_inspect_field`, `risk_first_inspect` | Some update accuracy or target accuracy can look good | Budgeted safe action, overinspection, cost-adjusted gates | Simple inspection heuristics are not cost-aware active inspection. |

# 6. Discussion

## 1. Probe false positive

A relation can be linearly readable but not behaviorally specific. The shortcut neural model retains readable relation information, but fails transfer, spurious attack, and causal specificity gates. The neural-to-table extraction result makes the same point: an extracted table can be editable while still being wrong or misaligned.

The newer neural stage adds a sharper distinction. `edit_pressure_training` can produce correct extracted tables and local table edits while failing to produce a stable causal linear relation subspace across seeds. Therefore, extracted-table editability and causal linear subspace structure should be treated as distinct diagnostics.

## 2. Prediction false positive

An agent can predict well without editability or auditability. This appears in the food-world memory/fitting baselines, the slope `structural_memory` baseline, and the temporal `structural_memory_temporal` baseline. The gates require counterfactual use, edits, audits, and relation alignment, not only success on sampled tasks.

## 3. Temporal memory false positive

An agent can remember temporal patterns without internal delayed links. V2/V2.1 show that delayed relation internalization requires temporal indexing, delay edits, and temporal audits. High temporal prediction is not sufficient.

## 4. Discovery false positive

An agent can discover relations but still act unsafely when the current relation chain is unverifiable. R2 is the clearest example: `discovery_relation_agent` reaches `0.967` partial observation success but receives a gated score of `0.000` because inspection recall, unsafe automation control, and uncertainty audit fail.

## 5. Inspection false positive

An agent can look safe by inspecting everything, but fail under cost and limited budget. R2.1 rejects `missing_always_inspect`; R3 rejects first-missing, random-field, and risk-first heuristics. The active inspection agent passes only when it selects informative fields, updates sequentially, avoids unsafe automation, and avoids overinspection.

# 7. Limitations

The claim boundary is strict.

- The environments are toy worlds.
- Variables are hand-specified.
- Several stages use predefined candidate relation graphs or declared nuisance priors.
- The uncertainty and inspection policies are rule-based.
- Data generators are deterministic or synthetic.
- Sensor noise and missingness are not real field processes.
- There is no real sensor reliability calibration.
- There is no adversarial missingness beyond the tested cases.
- There is no real slope mechanics.
- There is no deployment claim.
- The results do not establish real geotechnical safety capability.
- The results do not establish general causal discovery.
- The results do not establish object permanence.
- The results do not support claims that LLMs are replaced or that large neural systems naturally internalize relations.

# 8. Future Work

## Near-term

- Replace rule-based inspection value with learned uncertainty/value estimation while preserving the same gates.
- Add adversarial missingness and correlated sensor failures.
- Add calibration curves for confidence, not only gated pass/fail metrics.
- Connect neural-to-table extraction with active inspection.

## Medium-term

- Test whether neural policies can learn editable/auditable relation structures.
- Add learned sensor reliability.
- Scale to larger variable graphs while preserving counterfactual, edit, audit, uncertainty, and inspection gates.

## Long-term

- Move from toy worlds to validated simulators.
- Define what evidence would be required before any engineering safety claim.
- Build benchmark families combining distribution shift, intervention, temporal delay, partial observability, and inspection cost.

# 9. Final Claim

We present a staged toy diagnostic framework showing that relation internalization can be separated from prediction, probe readability, memory, surface shortcuts, and blanket inspection. In these environments, agents pass only when relation structure is usable for transfer, counterfactual action, edits, audits, uncertainty recognition, and cost-aware inspection.
