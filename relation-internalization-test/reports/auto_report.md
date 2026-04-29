# Auto Report: Relation Internalization Test

## Summary

```text
                      base_reward  base_success  reversal_adaptation_steps  reversal_post_reward  frozen_reversal_success  ood_success  spurious_robustness  spurious_resource_accuracy  counterfactual_accuracy  edit_success  edit_resource_success  edit_resource_locality  edit_locality  edit_reversal_success  relation_resource_accuracy  relation_shuffled_resource_accuracy  relation_shuffle_drop  relation_table_coverage  relation_table_accuracy  relation_table_alignment  internalization_score  gated_internalization_score
agent                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
robust_wide_relation        0.471         0.994                       52.6                 0.502                    0.138        1.000                1.000                       1.000                    1.000           1.0                    1.0                     1.0            1.0                  1.000                       1.000                                0.368                  0.632                      1.0                      1.0                       1.0                  0.913                        0.913
relation                    0.469         0.991                       53.2                 0.498                    0.138        1.000                1.000                       1.000                    1.000           1.0                    1.0                     1.0            1.0                  1.000                       1.000                                0.633                  0.367                      1.0                      1.0                       1.0                  0.888                        0.888
decision_tree               0.466         0.991                      300.0                -0.431                    0.138        0.773                0.550                       0.300                    0.400           0.0                    0.0                     1.0            1.0                  0.000                       0.400                                0.400                  0.000                      0.0                      0.0                       0.0                  0.248                        0.000
fitting                     0.472         0.996                       45.2                 0.518                    0.138        0.866                0.650                       0.325                    0.500           0.0                    0.0                     1.0            1.0                  0.000                       0.500                                0.500                  0.000                      0.0                      0.0                       0.0                  0.406                        0.000
memory                      0.471         0.990                      300.0                -0.437                    0.138        0.866                0.833                       0.500                    0.506           0.0                    0.0                     1.0            1.0                  0.000                       0.506                                0.506                  0.000                      0.0                      0.0                       0.0                  0.291                        0.000
majority                    0.003         0.586                      248.2                 0.004                    0.508        0.866                0.833                       0.500                    0.500           0.0                    0.0                     1.0            1.0                  0.000                       0.500                                0.500                  0.000                      0.0                      0.0                       0.0                  0.322                        0.000
predictive                  0.473         0.996                       28.0                 0.521                    0.138        0.898                0.608                       0.533                    0.667           0.0                    0.0                     1.0            1.0                  0.000                       0.667                                0.667                  0.000                      0.0                      0.0                       0.0                  0.426                        0.000
wide_relation               0.473         0.996                       51.0                 0.508                    0.138        1.000                0.500                       0.375                    0.840           1.0                    1.0                     1.0            1.0                  0.982                       0.840                                0.390                  0.450                      1.0                      1.0                       1.0                  0.836                        0.000
```

## Failure Checks

- FAIL: relation ood_success=1.000, best baseline=1.000
- FAIL: relation spurious_robustness=1.000, best baseline=1.000
- FAIL: relation spurious_resource_accuracy=1.000, best baseline=1.000
- FAIL: relation counterfactual_accuracy=1.000, best baseline=1.000
- FAIL: relation edit_success=1.000, best baseline=1.000
- FAIL: relation edit_resource_success=1.000, best baseline=1.000
- FAIL: relation edit_locality=1.000, best baseline=1.000
- FAIL: relation edit_resource_locality=1.000, best baseline=1.000
- FAIL: relation edit_reversal_success=1.000, best baseline=1.000
- FAIL: relation relation_shuffle_drop=0.367, best baseline=0.632
- FAIL: relation relation_table_alignment=1.000, best baseline=1.000
- FAIL: relation gated_internalization_score=0.888, best baseline=0.913

## Interpretation

This report is generated from the minimal experiment. Treat positive results as evidence for this testbed only, not as a broad theory claim.