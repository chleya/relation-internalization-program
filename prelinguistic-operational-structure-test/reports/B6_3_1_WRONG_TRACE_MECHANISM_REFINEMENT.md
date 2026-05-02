# B6.3.1 Wrong-Trace and Mechanism Necessity Refinement

## Purpose
B6.3.1 sharpens B6.3 blockers without entering B7 or making real-world safety claims.

## Key Results
- b63_1_policy_mean_score = 0.987
- wrong_trace_no_public_state_score = 1.000
- state_only_drop_on_wrong_trace_no_public_state = 0.550
- feedback_required_drop = 0.275
- history_required_drop = 0.550
- credit_buffer_required_drop = 0.550
- hidden_indirect_discovery_score = 1.000

## Interpretation
B6.3.1 is a diagnostic refinement. It may support narrower split-level mechanism evidence only where targeted ablations cause specific drops.

High aggregate score must not be interpreted as solved robustness. wrong_trace repair, feedback/history necessity, credit-buffer necessity, and hidden-indirect discovery remain toy diagnostics.

Hidden indirect discovery in this stage uses synthetic exploration/outcome-history cues. It is not real-world causal discovery.

B6.3.1 does not prove real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.
