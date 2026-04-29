# Neural Stage Report

## 1. Why This Stage Was Needed

Earlier R-series positive agents used explicit relation machinery: relation tables, editable links, audit interfaces, and inspection policies. That made the diagnostic clean, but left a reviewer concern:

```text
Is the project showing relation internalization, or just evaluating systems where relation structure was already written in?
```

This neural stage addresses that weakness by removing the hand-written positive relation agent. The models receive task pressure, counterfactual pressure, and edit pressure, then are evaluated for transfer, shortcut rejection, reversal adaptation, table extraction, table edit locality, and causal subspace intervention.

## 2. Models Compared

- `pure_prediction`: supervised MLP trained only for resource prediction.
- `prediction_bottleneck`: supervised MLP with a small bottleneck representation.
- `counterfactual_training`: MLP trained with nuisance-invariance and relation-counterfactual pressure.
- `edit_pressure_training`: support-conditioned model trained on relation edit episodes.
- `explicit_table_oracle`: upper bound with a hand-written table, not neural evidence.

## 3. Main Sweep Result

Mean over seeds:

| model | gated | OOD | shortcut | reversal | table | edit | locality | relation_drop | nuisance_drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure_prediction | 0.000 | 0.913 | 0.636 | 1.000 | 1.000 | 1.000 | 1.000 | 0.087 | 0.000 |
| prediction_bottleneck | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.507 | 0.507 |
| counterfactual_training | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.406 | 0.000 |
| edit_pressure_training | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.235 | 0.000 |
| explicit_table_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

The strongest current non-handwritten positive result is `counterfactual_training`, not `edit_pressure_training`.

## 4. Why Pure Prediction and Bottleneck Failed

`pure_prediction` achieves high train accuracy and high OOD accuracy, but fails shortcut rejection and relation-subspace causality. This supports the core thesis:

```text
prediction alone is not relation internalization.
```

`prediction_bottleneck` performs well on behavior and extraction metrics, but relation and nuisance subspace drops are both high (`0.507` and `0.507`). Compression does not produce a clean relation-specific causal subspace. It entangles relation and nuisance factors.

## 5. Why Counterfactual Training Is Strongest

`counterfactual_training` passes the configured gates on average:

- OOD: `1.000`
- shortcut rejection: `1.000`
- reversal adaptation: `1.000`
- counterfactual consistency: `1.000`
- table alignment: `1.000`
- table edit success: `1.000`
- edit locality: `1.000`
- relation subspace drop: `0.406`
- nuisance subspace drop: `0.000`
- gated score: `0.981`

This resolves part of the earlier hand-written-agent weakness. In this toy setting, a neural model without an explicit hand-written relation table can pass the relation-internalization gates when trained with counterfactual pressure.

The claim remains narrow: this is a toy neural diagnostic, not proof of general neural relation understanding.

## 6. Why Edit-Pressure Is Mixed

`edit_pressure_training` is not a simple failure. It achieves:

- OOD: `1.000`
- shortcut rejection: `1.000`
- reversal adaptation: `1.000`
- table alignment: `1.000`
- table edit success: `1.000`
- edit locality: `1.000`

However, it does not reliably pass relation-subspace intervention. Its mean relation-subspace drop is `0.235`, but this is driven by instability across seeds. The mean gated score is only `0.200`.

The correct interpretation is:

```text
edit_pressure_training produced partial relation-usable structure, but not a
stable linearly causal relation subspace under the current intervention test.
```

## 7. Failure Localization

V1.1 localizes the failure. Four out of five edit-pressure seeds fail only or primarily the `relation_subspace_drop` gate:

| seed | relation_subspace_drop | nuisance_subspace_drop | gated |
| --- | ---: | ---: | ---: |
| 0 | 0.188 | 0.000 | 0.000 |
| 1 | 0.138 | 0.000 | 0.000 |
| 2 | 0.528 | 0.000 | 1.000 |
| 3 | 0.144 | 0.000 | 0.000 |
| 4 | 0.176 | 0.000 | 0.000 |

Multi-site intervention shows:

| site | mean_relation_drop | mean_nuisance_drop |
| --- | ---: | ---: |
| query_hidden | 0.140 | 0.000 |
| support_relation_state | 0.000 | 0.000 |
| combined_hidden | 0.148 | 0.000 |
| post_edit_state | 0.000 | 0.000 |

The failure is not table extraction or edit locality. Those metrics are high. The failure is specifically the stability of a linearly causal relation subspace.

This suggests the relation signal may be distributed, nonlinear, episode-conditioned, query-dependent, or encoded in the interaction between query and relation state. The current linear subspace intervention may not fully capture that structure.

## 8. Claim Boundary

Supported:

- Pure prediction is insufficient in this toy setting.
- Bottleneck compression is insufficient because it entangles relation and nuisance subspaces.
- Counterfactual pressure can induce stable gated relation-internalized behavior in this toy setting.
- Edit-pressure training can produce extractable and editable relation behavior.
- Extracted-table editability and causal linear subspace structure are distinct diagnostics.

Unsupported:

- Edit-pressure training has been shown to reliably produce neural relation internalization.
- The edit-pressure model has a stable causal linear relation subspace across seeds.
- The result generalizes beyond this toy world.
- The result proves relation understanding in large neural models.
- The result proves unrestricted causal discovery.

## 9. Next Diagnostic If Needed

Do not tune edit-pressure just to pass the gate. If continuing, the next small diagnostic should test whether edit-pressure relation structure is nonlinear or interaction-based:

- compare linear probe, MLP probe, and decision-tree probe;
- shuffle query/support pairings;
- swap edit states across episodes;
- test whether behavior follows the swapped edit state.

This should be framed as diagnostic localization, not score chasing.
