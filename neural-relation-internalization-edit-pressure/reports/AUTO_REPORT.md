# Auto Report

## Results

| model | gated | ood | shortcut | reversal | table | edit | locality | rel_drop | nuis_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pure_prediction | 0.000 | 0.913 | 0.636 | 1.000 | 1.000 | 1.000 | 1.000 | 0.087 | 0.000 |
| prediction_bottleneck | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.507 | 0.507 |
| counterfactual_training | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.406 | 0.000 |
| edit_pressure_training | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.235 | 0.000 |
| explicit_table_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Interpretation

This is a toy diagnostic for neural relation internalization under edit pressure. A non-zero gated score means all configured gates passed; a zero score means at least one structural gate failed.

Current result:

- `pure_prediction` fails the gate despite high train accuracy.
- `prediction_bottleneck` fails because relation and nuisance subspace drops are not separated.
- `counterfactual_training` is the strongest neural result in this run and passes all gates on average.
- `edit_pressure_training` learns transfer, extraction, edit, and locality behavior, but only partially passes because relation-subspace intervention is not stable across seeds.
- `explicit_table_oracle` is an upper bound, not evidence of neural relation emergence.

This should be read as a mixed result: counterfactual pressure is sufficient in this toy implementation, while the current edit-pressure architecture needs a more stable relation-state/intervention design before it can be treated as the main positive result.
