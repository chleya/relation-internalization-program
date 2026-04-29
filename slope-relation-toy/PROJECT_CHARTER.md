# Project Charter: Slope Relation Toy

Date: 2026-04-28

## Purpose

Build a minimal slope-engineering toy world to test relation internalization in an engineering-review setting.

The project is not a real slope safety model. It is a diagnostic harness for this question:

```text
Can a system use an internal relation chain rather than surface warning labels?
```

## Core Relation Chain

```text
Rainfall -> Infiltration
Infiltration -> PorePressure
PorePressure -> Displacement
Displacement -> CrackExpansion
CrackExpansion -> RiskUp

Drainage -> PorePressureDown
Anchoring -> DisplacementDown
ToeExcavation -> StabilityDown
Monitoring -> UncertaintyDown
StopWork -> ExposureRiskDown
```

## Tests

- Base performance
- OOD surface-label transfer
- Spurious attack
- Counterfactual intervention
- Internal relation-link edit
- Relation-chain audit
- Irrelevant-link rejection
- Noisy monitoring observation test
- Engineering review score
- Review action-point consistency
- Gated slope score

## Baseline Risks Covered

```text
surface
```

Learns warning labels such as weather/report tokens instead of engineering relations.

```text
generic_review
```

Uses plausible engineering words such as drainage and support, but does not bind actions to relation-chain intervention points.

```text
structural_memory
```

Memorizes structural-variable contexts and can do well on action prediction, but has no editable relation chain or engineering review chain.

```text
learned_links
```

Learns a small editable relation-link table from observations. It is the first non-oracle relation-chain baseline.

Boundary:

```text
learned_links learns whether predefined candidate links are effective.
It does not discover an unrestricted engineering relation graph.
```

## Gate Logic

The gated score is zero unless:

```text
ood_success >= 0.8
spurious_attack_success >= 0.8
counterfactual_accuracy >= 0.8
edit_success >= 1.0
relation_audit >= 0.9
irrelevant_link_rejection >= 0.9
noisy_observation_success >= 0.7
review_score >= 0.9
review_consistency >= 0.9
```

## Claim Boundary

Supported:

- A relation-chain agent can be distinguished from surface-cue baselines in a slope-engineering toy world.
- OOD, spurious attack, counterfactual, edit, and audit gates prevent obvious false positives.

Not supported:

- Real engineering safety decisions.
- Numerical geotechnical simulation.
- Deployment as an AI review system.

## Self-Audit Requirement

Every run should keep `reports/self_audit.md` up to date.

The self-audit must state that `relation_chain` is an oracle-style hand-written model and that this project is not evidence of a deployable engineering AI system.
