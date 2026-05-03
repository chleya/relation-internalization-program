# B6.4.3 Dynamics-Delay-Credit Interaction Refinement Plan

## Purpose

B6.4.2 identified the main combined-remap failure source:

* combined_failure_source = dynamics_delay_credit_interaction
* combined_policy_score = 0.675
* combined_oracle_gap = 0.325

B6.4.3 should not enter B7.

B6.4.3 should diagnose and refine the interaction between dynamics remapping, delayed effects, credit assignment, and candidate search under combined hard remap.

## Core Question

Why does the policy fail when dynamics remap and delay-credit demands interact under combined hard remap?

## Candidate Failure Modes

1. delayed credit buffer assumes stable dynamics.
2. dynamics remap shifts the expected effect signature.
3. delayed outcome timing becomes ambiguous.
4. candidate search produces plausible but temporally shifted candidates.
5. dynamics-delay interaction causes wrong credit assignment.
6. policy cannot distinguish delayed success from delayed no-effect under remapped dynamics.
7. candidate search and credit buffer interact badly under shifted dynamics.

## Proposed Diagnostics

1. dynamics_delay_grid

Vary:

* dynamics shift magnitude
* delay steps
* stochastic delay variance
* effect signature noise

Measure:

* credit assignment accuracy
* delayed success recognition
* no-effect rejection
* backfire detection
* oracle gap

2. credit_buffer_adaptation_test

Test whether credit buffer can adapt expected effect signature under remapped dynamics.

Variants:

* fixed signature buffer
* adaptive signature buffer
* history-calibrated buffer
* oracle buffer baseline

3. candidate_search_temporal_alignment_test

Test whether candidate search selects paths with correct delayed timing.

Variants:

* spatial candidate correct but temporal signature wrong
* temporal candidate correct but spatial path noisy
* spurious candidate with correct short-term effect but wrong delayed effect

4. dynamics_delay_interaction_ablation

Disable:

* adaptive credit buffer
* candidate temporal filtering
* delayed outcome history
* dynamics-aware prediction
* inspection recovery

## Proposed Metrics

* dynamics_delay_credit_score
* credit_assignment_under_dynamics_shift
* adaptive_credit_buffer_gain
* temporal_alignment_accuracy
* delayed_success_vs_no_effect_under_dynamics_accuracy
* spurious_temporal_candidate_rejection_rate
* dynamics_delay_oracle_gap
* dynamics_delay_failure_source
* candidate_credit_interaction_score

## Definition of Done

B6.4.3 succeeds if it explains or reduces the dynamics_delay_credit_interaction failure.

Success does not require eliminating the oracle gap completely.

Success requires:

* clearer attribution of whether failure comes from dynamics shift, delay uncertainty, candidate temporal mismatch, or credit buffer rigidity.
* no leakage.
* no hidden labels.
* no real-world claim.
* no B7.

## Claim Boundary

B6.4.3 remains toy-to-toy diagnostic evidence.

It does not support:

* real-world transfer
* robotics capability
* safety certification
* construction-site autonomy
* deployable control
* B7 readiness unless gates are explicitly re-evaluated and passed.
