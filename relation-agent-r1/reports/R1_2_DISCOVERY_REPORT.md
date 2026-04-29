# R1.2 Discovery Report

## 1. Motivation

R1.2 reduces R1.1 scaffolding by replacing hand-written TRUE_LINK candidates with relation candidates enumerated from observed transitions.

## 2. Tests

- Unmarked nuisance rejection.
- New process link discovery outside the original TRUE_LINKS set.
- Adaptive exploration instead of a fixed exploration schedule.
- Discovered relation precision under nuisance pressure.
- Relation-guided action success.

## 3. Results

| agent | n | unmarked_nuisance_rejection | new_link_discovery | adaptive_exploration | discovered_relation_precision | discovery_action_success | discovery_r12_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 5 | 0.482 | 0.000 | 0.000 | 0.000 | 0.467 | 0.000 |
| shortcut | 5 | 0.915 | 0.000 | 0.000 | 0.000 | 0.918 | 0.000 |
| passive_memory | 5 | 0.773 | 0.000 | 0.000 | 0.000 | 0.788 | 0.000 |
| relation_agent | 5 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| discovery_relation_agent | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 4. Interpretation

The discovery agent must learn usable relations from transition evidence rather than receiving the original true-link table. Negative controls should still fail the gated score.

## 5. Boundary

R1.2 is not unrestricted causal discovery. It still uses a small process-variable schema, deterministic dynamics, and a toy action space.
