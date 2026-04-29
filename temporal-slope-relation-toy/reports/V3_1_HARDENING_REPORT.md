# V3.1 Takeover Hardening Report

## 1. Motivation

V3 introduced takeover under missing, noisy, delayed, and conflicting observations. V3.1 attacks a likely false positive: an agent may pass by taking over whenever any uncertainty appears, or by emitting generic audit text.

## 2. Attacks

- irrelevant missing sensor
- benign noise
- takeover overuse
- conflicting downstream evidence
- audit specificity

## 3. Gates

```text
{'irrelevant_missing_rejection': 0.9, 'benign_noise_rejection': 0.85, 'takeover_overuse_control': 0.85, 'conflicting_evidence_takeover': 0.9, 'audit_specificity': 0.85}
```

## 4. Results

```text
                                 seed  irrelevant_missing_rejection  benign_noise_rejection  takeover_overuse_control  conflicting_evidence_takeover  audit_specificity  hardening_v31_gated_score
agent                                                                                                                                                                                             
delayed_relation_chain            2.0                           1.0                     1.0                       1.0                            1.0              1.000                        1.0
instant_relation_chain            2.0                           1.0                     1.0                       1.0                            0.0              0.000                        0.0
learned_delayed_links             2.0                           1.0                     1.0                       1.0                            1.0              1.000                        1.0
overcautious_takeover             2.0                           0.0                     0.0                       0.0                            1.0              0.333                        0.0
structural_memory_temporal        2.0                           1.0                     1.0                       1.0                            0.0              0.000                        0.0
surface_temporal                  2.0                           1.0                     1.0                       1.0                            0.0              0.000                        0.0
uncertainty_aware_delayed_links   2.0                           1.0                     1.0                       1.0                            1.0              1.000                        1.0
```

## 5. Interpretation

Non-zero hardening_v31_gated_score requires rejecting irrelevant/benign uncertainty while still taking over for concrete relation-chain conflicts and naming the affected field/link in audit.

## 6. Claim Boundary

Supported:
- toy hardening against generic uncertainty takeover and template-only audit.

Unsupported:
- real engineering takeover policy.
- real monitoring safety model.
