# V3 Uncertainty and Takeover Report

## 1. Motivation

V2.1 showed that temporal prediction is not temporal relation internalization. V3 tests the next boundary: relation use under missing sensors, delayed noisy observations, and takeover conditions.

## 2. Setup

The toy temporal chain remains the V2/V2.1 delayed slope relation chain. V3 corrupts observations with missing values, noise, and delayed reports, then evaluates whether an agent should act automatically or trigger takeover.

## 3. Metrics

- noisy_action_success
- takeover_precision
- takeover_recall
- unsafe_automation_rate
- uncertain_relation_audit_score
- gated_v3_score

## 4. Gates

```text
{'noisy_action_success': 0.75, 'takeover_precision': 0.7, 'takeover_recall': 0.8, 'unsafe_automation_rate_max': 0.1, 'uncertain_relation_audit_score': 0.8}
```

## 5. Results

```text
                                 seed  noisy_action_success  takeover_precision  takeover_recall  unsafe_automation_rate  uncertain_relation_audit_score  gated_v3_score
agent                                                                                                                                                                   
delayed_relation_chain            2.0                 0.774               0.759            0.974                   0.026                             1.0           0.895
instant_relation_chain            2.0                 0.210               0.000            0.000                   1.000                             0.0           0.000
learned_delayed_links             2.0                 0.774               0.759            0.974                   0.026                             1.0           0.895
structural_memory_temporal        2.0                 0.416               0.734            0.093                   0.907                             0.0           0.000
surface_temporal                  2.0                 0.309               0.000            0.000                   1.000                             0.0           0.000
uncertainty_aware_delayed_links   2.0                 0.774               0.759            0.974                   0.026                             1.0           0.895
```

## 6. Interpretation

Non-zero gated_v3_score requires accurate action, high takeover precision/recall, low unsafe automation, and concrete uncertain relation audit.

## 7. Claim Boundary

Supported:
- toy diagnostic for uncertainty-aware temporal relation-chain takeover.

Unsupported:
- real geotechnical time-series modeling.
- deployment-ready engineering safety AI.
- unrestricted temporal relation discovery.
