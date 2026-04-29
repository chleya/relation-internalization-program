# V5.1 Governance Hardening Report

## 1. Motivation

V5.1 attacks governance-shell false positives.

## 2. Attacks

- fake replay hash
- approval gate label without enforcement
- responsibility boilerplate without human route
- missing relation evidence in logs
- route tampering after log creation

## 3. Gates

- `fake_replay_rejection` >= 0.9
- `gate_label_enforcement` >= 0.9
- `responsibility_route_consistency` >= 0.9
- `relation_evidence_preservation` >= 0.9
- `route_tamper_rejection` >= 0.9

## 4. Results

| shell | base_gated_v5_score | fake_replay_rejection | gate_label_enforcement | responsibility_route_consistency | relation_evidence_preservation | route_tamper_rejection | hardening_v51_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compliant_shell | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| auto_approve_shell | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| no_log_shell | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| no_replay_shell | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| no_responsibility_shell | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| fake_replay_shell | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| gate_label_only_shell | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| responsibility_boilerplate_shell | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| missing_relation_evidence_shell | 0.983 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| route_tampering_shell | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

## 5. Boundary

V5.1 is still a toy diagnostic, not real deployment governance.
