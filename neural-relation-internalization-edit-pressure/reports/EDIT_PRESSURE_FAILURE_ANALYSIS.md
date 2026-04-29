# Edit-Pressure Failure Analysis

## Question

Why does `edit_pressure_training` have strong transfer/table/edit/locality behavior but low mean gated score?

## Gate Failure Summary

- Seeds analyzed: 5.
- Seeds failing only or primarily the relation-subspace gate: 4.
- The per-seed gate matrix is in `results/gate_failure_matrix.csv`.

## Subspace Drop Distribution

| seed | relation_subspace_drop | nuisance_subspace_drop | gated |
| --- | ---: | ---: | ---: |
| 0 | 0.188 | 0.000 | 0.000 |
| 1 | 0.138 | 0.000 | 0.000 |
| 2 | 0.528 | 0.000 | 1.000 |
| 3 | 0.144 | 0.000 | 0.000 |
| 4 | 0.176 | 0.000 | 0.000 |

## Intervention Site Analysis

| site | mean_relation_drop | mean_nuisance_drop |
| --- | ---: | ---: |
| combined_hidden | 0.148 | 0.000 |
| post_edit_state | 0.000 | 0.000 |
| query_hidden | 0.140 | 0.000 |
| support_relation_state | 0.000 | 0.000 |

## Extraction/Subspace Correlation

| extraction_metric | correlation_with_relation_drop |
| --- | ---: |
| table_alignment | 0.000 |
| table_ood_accuracy | 0.000 |
| table_spurious_attack_accuracy | 0.000 |
| table_edit_success | 0.000 |
| edit_locality | 0.000 |

## Interpretation

`edit_pressure_training` should not be described as a stable success. It learns relation-usable behavior at the extracted-table level, but the causal linear relation-subspace signal is unstable across seeds under the current intervention test.

This supports the narrower interpretation: extracted-table editability and causal linear subspace structure are distinct diagnostics.

## Boundary

This analysis does not add a new experiment, does not change the model, and does not prove or disprove neural relation understanding. It localizes the current failure mode in a toy diagnostic.
