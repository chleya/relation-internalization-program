# Self Audit: Slope Relation Toy

Date: 2026-04-28

## Verdict

The project is useful as a relation-chain diagnostic, but it is not evidence of a deployable engineering AI system.

## Checks

- `relation_chain` gated score: 0.980
- `surface` gated score: 0.000
- `structural_memory` gated score: 0.000
- `learned_links` gated score: 0.980
- `generic_review` gated score: 0.000

## Strengths

- Surface-cue fitting is rejected by OOD and spurious-attack tests.
- Generic engineering wording is rejected by review scoring and action-point consistency.
- Structural memory is separated from editable/auditable relation-chain internalization.
- `learned_links` tests whether an induced editable relation table can pass the same gates.
- Irrelevant-link rejection checks that hand-picked distractor links remain disabled.
- Noisy-observation testing checks basic robustness to monitoring-field corruption.
- The relation-chain model supports counterfactual checks, relation-link edits, and relation audit.

## Known Weaknesses

- `relation_chain` is an oracle-style model: it shares the hand-written relation logic with the environment.
- `learned_links` learns only two predefined candidate links; it does not discover an unrestricted relation graph.
- The toy world is deterministic and far simpler than real slope engineering.
- Thresholds are manually chosen and should not be treated as calibrated safety thresholds.
- Monitoring noise is single-step field corruption; there is still no time dynamics or numerical geotechnical model.
- The noisy-observation gate is set to 0.7 because a 10% monitoring flip changes many optimal actions in this toy.
- `structural_memory` is stronger than surface cues, but it is still a table-memory baseline.

## False Positive Risks

- A model could memorize the fixed review-consistency cases.
- A model could pass by learning the small candidate-link set without learning broader engineering structure.
- A template could pass review fields if it encodes the same fixed action-point map.
- Passing this toy does not imply readiness for real plan approval or risk control.

## Next Required Improvement

Noisy monitoring is now single-step only.
Next: add time-dependent monitoring so learned links must handle delayed effects.

## Claim Boundary

Supported:

- The current harness distinguishes relation-chain reasoning from surface labels and generic review text.
- A minimal predefined-link learner can pass the same gates after observing enough samples.

Not supported:

- Autonomous discovery of slope engineering relations.
- Unrestricted relation-graph discovery.
- Real-world geotechnical correctness.
- Any safety-critical deployment claim.
