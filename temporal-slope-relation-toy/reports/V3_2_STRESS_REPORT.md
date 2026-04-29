# V3.2 Partial-Observability Stress Report

## 1. Motivation

V3.1 hardened takeover against generic overcaution. V3.2 stress-tests harder partial-observability cases: long missing spans, correlated sensor failure, drift-like conflicts, multi-conflict audit, and delayed response safety.

## 2. Stress Tests

- long missing sensor span
- correlated sensor failure
- drift conflict
- multi-conflict audit
- delayed response safety

## 3. Gates

```text
{'long_missing_takeover_recall': 0.9, 'correlated_failure_recall': 0.9, 'drift_conflict_takeover': 0.85, 'multi_conflict_audit_score': 0.85, 'delayed_response_safety': 0.85}
```

## 4. Results

```text
                                 seed  long_missing_takeover_recall  correlated_failure_recall  drift_conflict_takeover  multi_conflict_audit_score  delayed_response_safety  stress_v32_gated_score
agent                                                                                                                                                                                               
delayed_relation_chain            2.0                           1.0                        1.0                      1.0                         1.0                      1.0                     1.0
instant_relation_chain            2.0                           0.0                        0.0                      0.0                         0.0                      0.0                     0.0
learned_delayed_links             2.0                           1.0                        1.0                      1.0                         1.0                      1.0                     1.0
overcautious_takeover             2.0                           1.0                        1.0                      1.0                         0.0                      1.0                     0.0
structural_memory_temporal        2.0                           0.0                        0.0                      0.0                         0.0                      0.0                     0.0
surface_temporal                  2.0                           0.0                        0.0                      0.0                         0.0                      0.0                     0.0
uncertainty_aware_delayed_links   2.0                           1.0                        1.0                      1.0                         1.0                      1.0                     1.0
```

## 5. Interpretation

Non-zero stress_v32_gated_score requires takeover behavior to remain safe when observation uncertainty lasts across time or affects multiple related sensors.

## 6. Claim Boundary

Supported:
- toy stress testing for partial-observability relation-chain takeover.

Unsupported:
- real-world monitoring reliability.
- validated engineering takeover policy.
