# B2.1 Trace-Bearing Substrate Hardening

## 1. Purpose

B2 introduced trace-bearing substrates for delayed operational checkpoints. B2.1 tests whether these models truly use delayed causal trace.

## 2. Background

PLOS v1 found flow_checkpoint_model. B1.1 showed it fails delayed checkpoint. B2 introduced temporal/field/schema memory substrates. B2.1 attacks the trace itself.

## 3. Target Models

- recurrent_flow_checkpoint_model
- field_memory_model
- schema_memory_model

Optional reference:
- flow_checkpoint_model

## 4. Attack Set

- false delayed trace
- trace swap
- trace deletion specificity
- multi-source trace conflict
- noisy delayed trace
- trace length extrapolation
- trace compression pressure

## 5. Results Table

| model | seed | false_trace_rejection | false_trace_selected_rate | true_trace_selected_rate | trace_swap_sensitivity | trace_swap_endpoint_shift | trace_deletion_specificity_ratio | true_trace_intervention_drop | matched_non_trace_drop | non_trace_stability | multi_source_conflict_resolution | wrong_trace_follow_rate | noisy_trace_robustness | accuracy_under_mild_noise | accuracy_under_medium_noise | accuracy_under_strong_noise | noise_degradation_slope | trace_length_extrapolation | delay_8_accuracy | delay_10_accuracy | long_delay_degradation | trace_compression_survival | accuracy_at_075 | accuracy_at_050 | accuracy_at_025 | causal_trace_retention_under_compression | saliency_retention_bias | b21_trace_hardening_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 0.000 | 1.000 | 1.000 | 0.946 | 938847.382 | 0.939 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.950 | 1.000 | 0.900 | 0.100 | 0.667 | 1.000 | 1.000 | 0.000 | 0.667 | 0.000 | 0.959 |
| field_memory_model | 0 | 1.000 | 0.000 | 1.000 | 1.000 | 0.946 | 938847.382 | 0.939 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.950 | 1.000 | 0.900 | 0.100 | 0.667 | 1.000 | 1.000 | 0.000 | 0.667 | 0.000 | 0.959 |
| schema_memory_model | 0 | 1.000 | 0.000 | 1.000 | 1.000 | 0.946 | 938847.382 | 0.939 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.950 | 1.000 | 0.900 | 0.100 | 0.667 | 1.000 | 1.000 | 0.000 | 0.667 | 0.000 | 0.959 |

## 6. Interpretation

Temporal memory trace may be sufficient under B2.1, discounted by checkpoint and recurrent priors. Delayed operational trace may be field-form O, not object-form O. Sparse schema memory may be a strong delayed trace carrier, discounted by schema-slot priors.

If a model passes, it remains a hardened trace-bearing substrate candidate.
If it fails, its B2 success may rely on trace prior, clean-generator cues, or shallow delay shortcuts.

## 7. Claim Boundary

Do not claim blank-slate emergence.
Do not claim general physical reasoning.
Do not claim human-like trace cognition.
Do not claim real-world deployment.
