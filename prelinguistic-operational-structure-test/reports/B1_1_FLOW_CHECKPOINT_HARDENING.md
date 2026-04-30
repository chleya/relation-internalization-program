# B1.1 Flow-Checkpoint Reviewer Hardening

## 1. Purpose

Attack the current only PLOS candidate: flow_checkpoint_model.

## 2. Why This Is Needed

flow_checkpoint_model passed PLOS v1 but has a high checkpoint-selection prior.

## 3. Attack Set

- dynamic decoy checkpoint
- delayed checkpoint
- competing checkpoints
- checkpoint relocation OOD
- causal deletion vs visual deletion
- anti-prior world

## 4. Results Table

| metric | value | gate | pass |
| --- | ---: | ---: | --- |
| `candidate_gate_preserved` | 1.000 |  |  |
| `dynamic_decoy_rejection` | 0.843 | 0.800 | pass |
| `decoy_selected_rate` | 0.000 |  |  |
| `true_checkpoint_preservation` | 0.843 |  |  |
| `delayed_checkpoint_accuracy` | 0.000 | 0.750 | fail |
| `immediate_saliency_error_rate` | 0.000 |  |  |
| `delayed_endpoint_shift` | 0.000 |  |  |
| `competing_checkpoint_choice` | 0.970 | 0.750 | pass |
| `fixed_priority_error_rate` | 0.017 |  |  |
| `oracle_value_rank` | 1.063 |  |  |
| `relocation_ood_stability` | 0.840 | 0.750 | pass |
| `corner_checkpoint_accuracy` | 0.000 |  |  |
| `edge_checkpoint_accuracy` | 0.000 |  |  |
| `rare_position_accuracy` | 0.840 |  |  |
| `causal_deletion_shift` | 0.375 |  |  |
| `visual_deletion_shift` | 0.173 |  |  |
| `causal_over_visual_deletion_ratio` | 2.166 | 1.500 | pass |
| `anti_prior_survival` | 0.850 | 0.700 | pass |
| `prior_trap_selection_rate` | 0.083 |  |  |
| `anti_prior_critical_accuracy` | 0.850 |  |  |
| `causal_endpoint_shift` | 0.375 | 0.250 | pass |
| `b11_hardening_score` | 0.000 |  |  |

## 5. Interpretation

The current flow_checkpoint_model PLOS v1 pass is not robust under B1.1 attacks.

If passed, the flow-checkpoint substrate survives stronger reviewer attacks, but remains high-prior.
If failed, the prior PLOS pass was likely checkpoint-prior or saliency dependent.

## 6. Claim Boundary

Do not claim blank-slate emergence.
Do not claim general physical reasoning.
Do not claim real-world cognition.
Do not claim language-free intelligence solved.
