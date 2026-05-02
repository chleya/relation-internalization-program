# B6.3.1 Wrong-Trace and Mechanism Necessity Refinement Plan

## Purpose

B6.3.1 should target only the blockers exposed by B6.3. It should not introduce B7, a new world, robotics, engineering deployment, or real-world safety claims.

The goal is to sharpen causal diagnostic pressure around wrong trace repair, public-state dependence, feedback/history necessity, delayed credit buffer necessity, and hidden indirect causal path discovery.

## Core Blockers

1. wrong_trace remains explainable by state_only.
2. hide_public_state_cue aggregate drop = 0.000.
3. feedback update necessity is not proven.
4. history necessity is not proven.
5. delayed credit buffer necessity is weak in aggregate.
6. hidden indirect causal path discovery is not proven.

## A. Wrong-Trace Hardening

- Make wrong_trace impossible for state_only to solve perfectly.
- Reduce public state target leakage.
- Force repair to rely on inspect, history, and feedback where possible.
- Add wrong_trace_no_public_state.
- Add wrong_trace_history_conflict.
- Add wrong_trace_feedback_repair_required.
- Require wrong trace to trigger inspect, confidence downgrade, repair, or abstain.

## B. Public-State Dependency Test

- Hide or corrupt public state cues more aggressively.
- Compare b63_policy against state_only under hidden and corrupted public-state conditions.
- If b63_policy survives, identify whether history, feedback, inspection, or another public cue replaces direct state evidence.
- If state_only remains strong, downgrade the trace-repair claim.

## C. Feedback/History Necessity Sharpening

- Create conditions where feedback and history are required for correct repair or delayed decision.
- freeze_feedback_update should produce a large targeted drop in feedback-required splits.
- remove_history should produce a large targeted drop in history-required splits.
- If these ablations do not hurt, report that feedback/history are not necessary under the current diagnostic.

## D. Delayed Credit Buffer Necessity Sharpening

- Create delay5 cases where success cannot be scored without a credit buffer.
- disable_credit_buffer should collapse delayed credit success in those cases.
- Abstain and backfire avoidance must not count as delayed indirect success.
- no-effect and backfire outcomes must remain separate from successful delayed credit.

## E. Hidden Indirect Causal Path

- Hide indirect_target_region.
- Remove simple candidate shortcut where possible.
- Require discovery through outcome history, exploration, or trace-consistent indirect evidence.
- If the policy cannot discover the path without public target cue, report hidden indirect causal path discovery as unresolved.

## F. Evaluation Criteria

- Do not optimize mean score.
- Require split-level drops under targeted ablations.
- Require baseline separation from state_only, mask_only, trace_only, conservative_uncertainty, random, always_abstain, and oracle.
- Require no evaluator-ground-truth or oracle leakage.
- Require no empty metric success.
- Require score caps when unsafe action, false-safe commit, or invalid no-sample condition occurs.

## Definition of Done

- wrong_trace is no longer solved by state_only alone, or the report explicitly states that it still is.
- feedback/history necessity is either demonstrated in targeted splits or explicitly rejected.
- delayed credit buffer necessity is either demonstrated in targeted delay5 splits or explicitly rejected.
- hidden indirect path discovery is either improved without public target cue or explicitly remains unsolved.
- reports preserve all caveats and do not claim real-world risk intelligence, robotics ability, safety certification, construction-site autonomy, or deployable engineering control.

