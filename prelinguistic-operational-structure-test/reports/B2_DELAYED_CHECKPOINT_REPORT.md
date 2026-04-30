# B2 Delayed Operational Checkpoint Substrate

## 1. Purpose

B1.1 found that flow_checkpoint_model fails delayed_checkpoint_accuracy. B2 tests whether delayed operational checkpoint requires memory, field trace, or schema memory substrate.

## 2. Models

- flow_checkpoint_model
- recurrent_flow_checkpoint_model
- field_memory_model
- schema_memory_model

## 3. Tasks

- delayed checkpoint
- multi-delay checkpoint
- early saliency rejection
- causal trace intervention
- delay OOD

## 4. Results

| model | seed | delayed_checkpoint_accuracy | multi_delay_stability | early_saliency_rejection | delay_ood_generalization | causal_trace_intervention_drop | non_trace_stability | delayed_endpoint_shift | trace_over_control_ratio | b2_delayed_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flow_checkpoint_model | 0 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.954 | 1.000 | 0.954 | 954109.405 | 1.000 |
| field_memory_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.954 | 1.000 | 0.954 | 954109.405 | 1.000 |
| schema_memory_model | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.954 | 1.000 | 0.954 | 954109.405 | 1.000 |

## 5. Interpretation

Temporal memory may be sufficient to extend checkpoint substrate from short-horizon to delayed checkpoint. Delayed operational structure may be field-trace form, supporting a non-object-centric delayed O candidate. Sparse schema memory may be a stronger substrate for delayed operational checkpoints. flow_checkpoint_model remains short-horizon, confirming the B1.1 diagnosis.

## 6. Failure Analysis

If all fail: delayed checkpoint remains unresolved.
If only recurrent_flow succeeds: temporal memory may be sufficient, with flow-checkpoint prior still discounted.
If field_memory succeeds: delayed O may be field-trace form.
If schema_memory succeeds: sparse delayed schema may be strongest candidate, with injected structure prior discounted.
If flow_checkpoint still fails: this confirms the B1.1 short-horizon limitation.

## 7. Claim Boundary

Do not claim general physical reasoning.
Do not claim blank-slate emergence.
Do not claim B-line solved.
