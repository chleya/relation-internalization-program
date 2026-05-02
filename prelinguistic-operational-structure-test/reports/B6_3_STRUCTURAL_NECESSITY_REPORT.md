# B6.3 Structural Necessity Report

## Purpose
B6.3 tests whether B6.2 behavior causally depends on internal operational trace, history, feedback, delayed credit, and candidate-search structure.

## Why B6.3
B6.2 still had unresolved wrong_trace behavior, candidate-search-only hide_indirect_target, and unproven private trace necessity.

## Ablation Table
| ablation | mean observed drop | interpretation |
| --- | ---: | --- |
| corrupt_trace | 0.124 | evidence_for_necessity |
| disable_candidate_search | 0.127 | evidence_for_necessity |
| disable_delayed_credit_buffer | 0.006 | weak_or_no_necessity_evidence |
| disable_inspection_recovery | 0.095 | weak_or_no_necessity_evidence |
| freeze_feedback_update | 0.019 | weak_or_no_necessity_evidence |
| remove_history | 0.025 | weak_or_no_necessity_evidence |
| remove_risk_cue | 0.179 | evidence_for_necessity |
| remove_trace | 0.124 | evidence_for_necessity |
| shuffle_trace | 0.124 | evidence_for_necessity |

## Claim Boundary
B6.3 can only support toy diagnostic structural necessity evidence if targeted ablations show specific performance collapses. It does not support real-world risk intelligence, safety certification, robotics ability, construction-site autonomy, or engineering deployment.
