# Neural Stage Final Summary

## 1. Why This Stage Was Needed

Earlier R-series diagnostics relied partly on hand-designed relation agents, relation tables, edit interfaces, audit structures, or inspection policies. Those agents made the diagnostic criteria explicit, but they also left a reviewer concern: positive evidence might be coming from relation machinery written into the agent rather than from relation-internalization-like structure learned by a neural model.

This neural stage tests whether relation-internalization-like structure can emerge under training pressure. The stage keeps the food/poison toy world, but compares neural models trained with prediction, compression, counterfactual, and edit pressures, plus an explicit table oracle as an upper bound.

## 2. Models Compared

- `pure_prediction`: a supervised MLP trained only to predict resource labels.
- `prediction_bottleneck`: a supervised MLP with a small bottleneck representation.
- `counterfactual_training`: an MLP trained with nuisance-invariance and relation-counterfactual losses.
- `edit_pressure_training`: a support-conditioned model trained on relation edit episodes.
- `explicit_table_oracle`: a hand-written relation table upper bound, not neural evidence.

## 3. Main Sweep Results

| model | gated_internalization_score | main interpretation |
| --- | ---: | --- |
| `pure_prediction` | 0.000 | Prediction alone does not pass the relation-internalization gate. |
| `prediction_bottleneck` | 0.000 | Compression alone does not separate relation and nuisance structure. |
| `counterfactual_training` | ~0.981 | Strongest current non-handwritten neural positive result. |
| `edit_pressure_training` | ~0.200 | Mixed: table-level editability and edit responsiveness, but unstable causal relation evidence. |
| `explicit_table_oracle` | 1.000 | Upper bound with hand-written table, not evidence of neural emergence. |

## 4. Why Pure Prediction Failed

Pure prediction may learn the training distribution and can exploit correlations that are useful for ordinary accuracy. It does not provide sufficient evidence of relation internalization under the full gate, because the gate requires OOD transfer, shortcut rejection, relation-subspace specificity, table extraction, table editability, and edit locality. In this run, `pure_prediction` has high ordinary performance but receives a gated score of `0.000`.

## 5. Why Bottleneck Failed

Compression alone does not guarantee relation structure. The bottleneck model can perform well on behavioral metrics while still mixing relation and nuisance information. In the current run, `prediction_bottleneck` fails because relation and nuisance subspace drops are both high, so the representation is not relation-specific.

## 6. Why Counterfactual Training Is the Strongest Neural Positive Result

Counterfactual training directly pressures two behaviors: invariance to nuisance variables and sensitivity to relation variables. In the current toy implementation, it is the most stable neural model under the gated internalization standard. It passes the configured gates on average with a gated score around `0.981`, without using a hand-written relation table as its positive mechanism.

This remains a toy result. The supported wording is relation-internalization-like behavior under counterfactual pressure, not general neural relation understanding.

## 7. Why Edit-Pressure Is Mixed

Edit-pressure training supports extracted-table behavior, edit/locality behavior, and edit-state responsiveness. However, it does not reliably pass stable `relation_subspace_drop` across seeds.

Edit-pressure training produces a behaviorally effective edit pathway, but not robust support-conditioned relation internalization.

This is a useful negative result. It shows that table-level editability and edit-signal responsiveness can look strong while support-conditioned relation binding and stable causal relation subspaces remain weak.

## 8. V1.1 Failure Localization

V1.1 localizes the `edit_pressure_training` failure:

- Four out of five seeds mainly fail the `relation_subspace_drop` gate.
- Only seed 2 clearly passes `relation_subspace_drop`.

| seed | relation_subspace_drop | gate |
| --- | ---: | --- |
| 0 | 0.188 | fail |
| 1 | 0.138 | fail |
| 2 | 0.528 | pass |
| 3 | 0.144 | fail |
| 4 | 0.176 | fail |

Multi-site intervention averages:

| site | mean relation_drop |
| --- | ---: |
| `query_hidden` | ~0.140 |
| `support_relation_state` | ~0.000 |
| `combined_hidden` | ~0.148 |
| `post_edit_state` | ~0.000 |

The model may not contain a stable linear causal relation subspace, even when extracted-table editability looks good.

## 9. V1.2 Interaction Diagnostics

V1.2 tests whether the missing causal relation evidence is instead nonlinear or interaction-mediated. The key results for `edit_pressure_training` are:

- `edit_state_swap_success = 1.000`
- `support_shuffle_drop = 0.000`
- `support_conditioned_accuracy = 0.500`
- `binding_sensitivity = 0.000`
- nonlinear probes show no extra nonlinear relation gain; linear and MLP probes are already readable.

The edit pathway has causal behavioral effect, but support-query relation binding is weak. V1.2 does not support the claim that strong interaction-mediated relation structure has formed.

## 10. Final Neural Stage Claim

In this toy neural setting, counterfactual training provides the strongest non-handwritten evidence for relation-internalization-like structure. Edit-pressure training exposes a distinct false positive: a model can respond to explicit edit signals and support table-level edits without robust support-conditioned relation internalization or stable causal relation subspaces.

## 11. What This Stage Adds to the Overall Project

This stage converts the previous diagnostic chain from "what does not count as relation internalization" toward a neural question: which training pressures can induce relation-internalization-like structure without hand-written relation agents?

The main answer is conservative. Counterfactual pressure is currently the strongest positive neural condition. Edit pressure is not a stable success; instead, it reveals that editable behavior itself can be a false positive.

## 12. Claim Boundary

Supported:

- Pure prediction and bottleneck are insufficient in this toy setting.
- Counterfactual pressure can induce stronger neural relation-internalization-like behavior.
- Edit-state responsiveness and table-level editability are not sufficient evidence of support-conditioned relation internalization.
- Causal linear subspace structure and table editability are separable diagnostics.

Unsupported:

- general neural relation understanding
- large-model behavior
- unrestricted causal discovery
- proof that edit-pressure is sufficient
- real-world engineering reasoning
- deployment-ready AI
