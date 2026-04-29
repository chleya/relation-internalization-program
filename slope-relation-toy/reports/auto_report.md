# Auto Report: Slope Relation Toy

Mean over seeds:

```text
                   ood_success  spurious_attack_success  counterfactual_accuracy  edit_success  relation_audit  irrelevant_link_rejection  noisy_observation_success  review_score  review_consistency  gated_slope_score
agent                                                                                                                                                                                                                    
generic_review           0.193                    0.162                    0.333           0.0             0.0                        1.0                      0.192         0.429                 0.0               0.00
learned_links            1.000                    1.000                    1.000           1.0             1.0                        1.0                      0.753         1.000                 1.0               0.98
majority                 0.571                    0.581                    0.667           0.0             0.0                        1.0                      0.570         0.000                 0.0               0.00
relation_chain           1.000                    1.000                    1.000           1.0             1.0                        1.0                      0.753         1.000                 1.0               0.98
structural_memory        1.000                    1.000                    1.000           0.0             0.0                        1.0                      0.976         0.000                 0.0               0.00
surface                  0.327                    0.118                    0.333           0.0             0.0                        1.0                      0.312         0.000                 0.0               0.00
```

Claim boundary: this is a toy relation-chain diagnostic, not a real slope safety model.