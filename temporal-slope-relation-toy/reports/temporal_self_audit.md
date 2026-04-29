# Temporal Self Audit

This is not a real geotechnical time-series model.
It only tests delayed relation-chain internalization in a deterministic toy world.

## Mean Scores

```text
                            seed  temporal_ood_success  surface_shortcut_rejection  delayed_counterfactual_accuracy  delay_edit_success  temporal_audit_score  gated_temporal_score
agent                                                                                                                                                                              
delayed_relation_chain       2.0                 1.000                       1.000                              1.0                 1.0                   1.0                   1.0
instant_relation_chain       2.0                 0.412                       0.369                              0.0                 0.0                   0.0                   0.0
learned_delayed_links        2.0                 1.000                       1.000                              1.0                 1.0                   1.0                   1.0
structural_memory_temporal   2.0                 1.000                       0.967                              1.0                 0.0                   0.0                   0.0
surface_temporal             2.0                 0.264                       0.261                              0.0                 0.0                   0.0                   0.0
```

## Known Weaknesses

- Delays are predefined candidates, not unrestricted discovery.
- Sequence length is short.
- State variables are directly observable in several baselines.
- No calibrated safety threshold is claimed.
- No real monitoring data is used.