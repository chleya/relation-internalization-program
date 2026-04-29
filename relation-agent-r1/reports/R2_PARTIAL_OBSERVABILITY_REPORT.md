# R2 Partial Observability Report

## 1. Motivation

R2 tests whether an active relation agent can use learned relations when observations are missing, noisy, or conflicting.

## 2. Tests

- Partial observation success.
- Inspection recall when critical relation nodes are unknown.
- Unsafe automation rate.
- Relation-specific uncertainty audit.
- Noisy observation robustness.

## 3. Results

| agent | n | partial_observation_success | inspection_recall | unsafe_action_rate | uncertainty_audit_score | noisy_observation_robustness | partial_r2_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 5 | 0.433 | 0.160 | 0.840 | 0.000 | 0.325 | 0.000 |
| shortcut | 5 | 0.893 | 0.000 | 1.000 | 0.000 | 0.470 | 0.000 |
| passive_memory | 5 | 0.772 | 0.000 | 1.000 | 0.000 | 0.412 | 0.000 |
| discovery_relation_agent | 5 | 0.967 | 0.000 | 1.000 | 0.000 | 0.448 | 0.000 |
| uncertainty_discovery_agent | 5 | 1.000 | 1.000 | 0.000 | 1.000 | 0.912 | 0.982 |

## 4. Interpretation

A passing agent must inspect before acting when key relation links cannot be verified, then act from the revealed state. Prediction or memory alone is not enough.

## 5. Boundary

R2 remains a toy diagnostic. It is not a real monitoring system, not engineering safety automation, and not unrestricted causal discovery.
