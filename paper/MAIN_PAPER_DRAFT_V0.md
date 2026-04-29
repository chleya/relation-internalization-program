# When Prediction Is Not Relation Internalization:
# Editable and Auditable Diagnostics for Relational Agents

# Abstract

When should an agent be credited with relation internalization rather than merely predicting outcomes, memorizing contexts, following shortcuts, producing plausible review text, or responding to explicit edit signals? We study this question with a staged chain of controlled toy diagnostics. Each stage introduces false-positive baselines that can look strong under ordinary metrics but fail relation-internalization gates.

The central thesis is conservative: task success, prediction accuracy, probe readability, memory, surface shortcuts, plausible review text, temporal prediction, blanket inspection, and edit-signal responsiveness are insufficient evidence of relation internalization. The supported claim is narrower: in controlled toy diagnostic environments, relation internalization can be operationalized as relation structures that are usable for transfer, counterfactual action, edits, audits, uncertainty recognition, and cost-aware inspection.

# 1. Introduction

Ordinary task accuracy is too weak for claims about relation internalization. A policy can succeed by exploiting shortcut variables, memorizing contexts, using structural memory, producing plausible language, predicting delayed outcomes, inspecting everything, or responding to an explicit edit signal without binding behavior to an inferred relation.

This paper treats relation internalization as an operational diagnostic standard, not as a theory of intelligence. The question is not whether a model "understands" relations. The question is whether relation structure is usable in ways relation claims require: transfer, counterfactual action, intervention/edit, audit, temporal indexing, uncertainty handling, and cost-aware inspection.

Every stage was designed around a baseline that should pass if ordinary performance were enough, but fails once relation internalization is required.

# 2. Operational Definition

We define relation internalization as:

```text
External relations become internal usable structures.
```

Usable means:

- transfer across distribution shift;
- counterfactual use;
- local intervention/edit;
- audit of relation links;
- action guidance;
- temporal indexing when relation effects are delayed;
- uncertainty recognition under partial observability;
- cost-aware inspection under budget.

This standard distinguishes relation internalization from:

- high prediction accuracy;
- bottleneck compression;
- probe readability;
- context memory;
- surface cue use;
- generic review language;
- temporal memory;
- blanket inspection;
- edit-signal responsiveness without support-conditioned binding.

# 3. False-Positive Ladder

The diagnostic ladder is organized around false positives:

| false positive | why it can look good | why it is insufficient |
| --- | --- | --- |
| prediction success | high train/OOD accuracy | may rely on shortcuts or memorized mappings |
| bottleneck compression | compact representations | can entangle relation and nuisance factors |
| probe readability | relation labels linearly decodable | readable information may not be behaviorally causal |
| structural memory | high success on known patterns | lacks editable/auditable relation links |
| generic review text | plausible explanation strings | language can be detached from relation behavior |
| temporal memory | delayed predictions succeed | lacks editable/auditable temporal links |
| fixed-delay template | passes one delay setting | fails variable delay and delay-edit tests |
| relation discovery without uncertainty | high action success under partial inputs | can automate unsafely when the chain is unverifiable |
| blanket inspection | appears conservative | fails precision, cost, and budget constraints |
| first-missing/random/risk-first inspection | simple heuristics can sometimes work | do not target highest relation value under budget |
| edit-signal responsiveness | edit swaps change outputs | may not bind query behavior to support-inferred relations |

# 4. Experimental Groups

## 4.1 Static Relation Diagnostics

The static line includes the food-world explicit relation table and the R1/R1.1/R1.2 active relation agent line. These tests require transfer, OOD action, relation recovery, counterfactual accuracy, edit success, active exploration, candidate discovery, and rejection of hidden confounders or irrelevant candidates.

Purpose: rule out memory, fitting, shortcut policies, no-exploration relation tables, and predefined `TRUE_LINK` dependence.

## 4.2 Neural Relation Diagnostics

The neural line includes the hidden-state relation probe, causal subspace intervention, neural-to-table extraction, and the newer counterfactual/edit-pressure stage.

The current neural stage compares:

- `pure_prediction`;
- `prediction_bottleneck`;
- `counterfactual_training`;
- `edit_pressure_training`;
- `explicit_table_oracle`.

The main update is that `counterfactual_training` is the strongest current non-handwritten neural positive result. `edit_pressure_training` is mixed: it shows table-level editability and edit-state responsiveness, but not strong support-conditioned relation binding or stable causal relation subspaces. Therefore, editable behavior itself must be treated as a potential false positive.

## 4.3 Engineering-Style Relation Chains

The slope-relation toy evaluates relation-chain agents against `structural_memory`, `generic_review`, `surface`, and other baselines. It tests OOD transfer, spurious attack robustness, counterfactuals, edits, relation audit, irrelevant-link rejection, noisy observations, and review consistency.

Purpose: rule out structural memory, surface labels, and plausible review language as sufficient evidence.

## 4.4 Temporal Relation Diagnostics

Temporal V2 and V2.1 test delayed relation chains. V2.1 hardens the setting with variable delays, false delay shortcut rejection, multi-link delay edits, temporal audit consistency, and anti-template generalization.

Purpose: rule out same-step logic, fixed-delay templates, temporal memory without editable delay links, false temporal shortcuts, and audit strings without real time indexes.

## 4.5 Partial Observability and Active Inspection

R2 tests partial observability; R2.1 tests relation-specific uncertainty versus blanket inspection; R3 tests active inspection selection under cost and budget. The required behavior is not merely to inspect, but to select informative fields, update sequentially, avoid unsafe automation, and avoid overinspection.

Purpose: rule out relation discovery without uncertainty handling, blanket inspection, first-missing/random/risk-first heuristics, and unsafe automation under unverifiable relation chains.

# 5. Results Summary

| group | positive result | key negative controls | result anchor |
| --- | --- | --- | --- |
| Neural probe/extraction | base neural model and base extracted table pass gates | shortcut neural model | base gated neural score about `0.819`; shortcut gated score `0.000`; base extraction `1.000`; shortcut extraction `0.000` |
| Neural counterfactual/edit pressure | `counterfactual_training` | `pure_prediction`, `prediction_bottleneck`, `edit_pressure_training` as mixed false positive | `counterfactual_training` gated `~0.981`; edit-pressure gated `~0.200` |
| Slope toy | `relation_chain` / `learned_links` | `structural_memory`, `generic_review`, `surface` | relation-chain/learned-links gated about `0.98`; structural/generic controls `0.000` |
| Temporal V2/V2.1 | `delayed_relation_chain`, `learned_delayed_links` | `structural_memory_temporal`, `instant_relation_chain` | delayed-link hardening `1.000`; structural temporal and instant controls `0.000` |
| R1/R1.1/R1.2 | active/discovery relation agents | random, shortcut, passive, no-explore, hand-supplied candidate dependence | R1 `0.972`; R1.1 `1.000`; R1.2 discovery `1.000` |
| R2/R2.1/R3 | uncertainty and active inspection agents | discovery-only, missing-always, first-missing, random, risk-first | R2 uncertainty `0.982`; R2.1 relation-specific `0.933`; R3 active inspection `0.933`; baselines `0.000` |

The important neural update is the editability false positive. V1.2 shows `edit_state_swap_success = 1.000`, but `support_shuffle_drop = 0.000`, `support_conditioned_accuracy = 0.500`, and `binding_sensitivity = 0.000`. Edit-signal responsiveness is not enough.

# 6. What Each Stage Rules Out

| Stage | Positive agent/result | Negative control | Ordinary metric that could look good | Gate that fails | Alternative explanation ruled out |
| --- | --- | --- | --- | --- | --- |
| Neural probe | base neural model | shortcut neural model | probe readability | OOD/spurious/subspace gates | readable relation information is not enough |
| Neural training pressure | `counterfactual_training` | `pure_prediction` | train/OOD accuracy | shortcut and relation-subspace gates | prediction success is not enough |
| Neural training pressure | `counterfactual_training` | `prediction_bottleneck` | behavior and extraction metrics | nuisance-subspace gate | compression can entangle factors |
| Neural edit pressure | mixed `edit_pressure_training` | edit-signal responsive behavior | table edit/locality | support binding and relation-subspace diagnostics | editable behavior can be a false positive |
| Slope toy | `relation_chain` / `learned_links` | `structural_memory` | OOD/spurious success | edit/audit/review gates | structural memory is not relation structure |
| Slope toy | relation-chain agents | `generic_review` | plausible text | relation behavior gates | review text is not relation use |
| Temporal V2/V2.1 | delayed-link agents | `structural_memory_temporal` | temporal prediction | delay edit/audit gates | temporal memory is not delayed relation structure |
| Temporal V2/V2.1 | delayed-link agents | `instant_relation_chain` | same-step relation logic | variable delay and edit gates | immediate logic cannot replace temporal indexing |
| R2 | `uncertainty_discovery_agent` | `discovery_relation_agent` | partial observation success `0.967` | uncertainty/unsafe automation gates | discovery alone is not enough |
| R2.1 | relation-specific uncertainty | `missing_always_inspect` | high critical recall | precision/cost gates | blanket inspection is not uncertainty handling |
| R3 | active inspection agent | first/random/risk-first baselines | some inspection success | target/cost/budget gates | simple inspection heuristics are not cost-aware selection |

# 7. Discussion

The paper is best read as a false-positive analysis.

Probe readability is insufficient because a relation can be decodable without being behaviorally causal. Prediction is insufficient because success can come from shortcuts, memory, or structural correlation. Review text is insufficient because an explanation can be plausible while disconnected from relation use. Temporal prediction is insufficient because delayed links must be editable and auditable. Discovery is insufficient because a discovered relation may be unverifiable in the current observation. Inspection is insufficient because blanket conservatism fails under cost and budget.

The neural stage adds a final caution: editable behavior is not necessarily relation internalization. `edit_pressure_training` can respond to edit signals and support table-level edits without robust support-conditioned binding or stable causal relation subspaces. The strongest current neural positive result is therefore `counterfactual_training`, not edit pressure.

# 8. Limitations

- All environments are toy diagnostics.
- Variables are hand-specified.
- Some stages use predefined candidate relation graphs.
- Uncertainty and inspection policies are rule-based.
- Data generators are synthetic.
- The neural setting is small and controlled.
- Counterfactual training may be viewed as supervised shortcut control, not evidence of broad emergence.
- No real slope mechanics are modeled.
- No real sensor reliability calibration is performed.
- No adversarial missingness robustness is claimed beyond tested toy cases.
- No real engineering safety or deployment claim is made.
- No claim is made about large language models or general causal discovery.

# 9. Future Work

Near-term:

- Replace rule-based inspection value with learned uncertainty/value estimation while preserving gates.
- Add adversarial missingness and correlated sensor failures.
- Add confidence calibration curves.
- Connect neural-to-table extraction with active inspection.
- Test whether counterfactual-trained neural structures remain editable under larger relation graphs.

Medium-term:

- Train neural policies to learn editable/auditable relation structures without hand-written relation agents.
- Learn sensor reliability instead of hand-specifying it.
- Scale to larger variable graphs and more complex temporal dependencies.

Long-term:

- Move from toy worlds to validated simulators.
- Define evidence required before any engineering safety claim.
- Build benchmark families combining distribution shift, intervention, temporal delay, partial observability, and inspection cost.

# 10. Final Claim

We present a staged toy diagnostic framework showing that relation internalization can be separated from prediction, probe readability, memory, surface shortcuts, review-like text, temporal prediction, blanket inspection, and edit-signal responsiveness. In these environments, agents pass diagnostic gates only when relation structure is usable for transfer, counterfactual action, edits, audits, uncertainty recognition, and cost-aware inspection.
