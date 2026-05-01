# PLOS-Test

Full name: Pre-Linguistic Operational Structure Test

This is a B-line project in the Relation Internalization Program.

It is not a generic video prediction benchmark. It is not an LLM task. It does
not test language fluency, text audit, or symbolic relation tables.

## Core Path

```text
W -> O1 -> O2 -> L
```

```text
W  = world process / fact-occurrence layer.
O1 = trajectory schema layer.
O2 = operational structure layer.
L  = language overlay.
```

Facts happen before language. Language can later cover, express, transmit,
audit, and reshape operational structure, but language fluency is not evidence
of operational structure. This project tests whether operational structure can
emerge from non-linguistic continuous dynamics.

The v1 target is:

```text
W -> O1 -> O2
```

Language belongs to later `O2 -> L` tests and is deliberately excluded here.

## Research Question

Can a model form operational structure from continuous 2D dynamics without
language labels, and can that structure be verified by behavior, structural
intervention, and OOD generalization?

## Evidence Rule

```text
Behavior + Structural Intervention + OOD
```

Behavior alone is not enough. Prediction accuracy alone is not enough. Tracking
accuracy alone is not enough. Inspection accuracy alone is not enough.

A model can only be called a PLOS candidate if:

- it passes core behavior gates;
- it passes at least one relevant structural intervention gate;
- it passes OOD gates;
- the intervention effect is local, predictable, and task-aligned;
- it does not rely on language labels or explicit symbolic relation tables.

## Main Candidate

The main candidate substrate is the Field-First Hybrid Schema Model:

```text
past frames
  -> latent dynamic field
  -> velocity / force / uncertainty / inspection-value fields
  -> sparse schema readouts
  -> event boundary map / critical region map / relation locality map
  -> prediction / intervention / inspect outputs
```

This choice avoids assuming object slots too early while still exposing
intervenable operational structures.

## Branches

```text
pixel_predictor       first-order frame prediction baseline
predictive_coding_model prediction-error substrate candidate
trajectory_memory     nearest-neighbor trajectory fragment baseline
patch_graph_model     local patch dynamic graph substrate candidate
koopman_model         low-rank dynamics substrate candidate
world_model           latent sequence predictor baseline
slot_model            object-biased second-order candidate
field_model           non-object-centric second-order candidate
flow_checkpoint_model trajectory-flow checkpoint substrate candidate
schema_model          field-first hybrid schema main candidate
```

## Run

```powershell
pip install -r requirements.txt
pytest -q
python -m src.run_sweep --config configs/sweep.yaml
python -m src.run_hardening --config configs/sweep.yaml --seed 0
python -m src.run_b11_flow_hardening --config configs/b11_flow_hardening.yaml --seed 0
python -m src.run_b2_delayed_checkpoint --config configs/b2_delayed_checkpoint.yaml --seed 0
python -m src.run_b21_trace_hardening --config configs/b21_trace_hardening.yaml --seed 0
python -m src.run_b21a_degeneracy_audit --config configs/b21a_degeneracy_audit.yaml --seed 0
python -m src.run_b22_selector_disentanglement --config configs/b22_selector_disentanglement.yaml --seed 0
python -m src.run_b23_private_selector --config configs/b23_private_selector.yaml --seed 0
python -m src.run_b3_active_inspection --config configs/b3_active_inspection.yaml --seed 0
python -m src.run_b31_inspection_audit --config configs/b31_inspection_audit.yaml --seed 0
python -m src.run_b32_mechanism_inspection --config configs/b32_mechanism_inspection.yaml --seed 0
python -m src.run_b4_intervention --config configs/b4_intervention.yaml --seed 0
python -m src.run_b41_intervention_audit --config configs/b41_intervention_audit.yaml --seed 0
python -m src.run_b42_action_type_disambiguation --config configs/b42_action_type_disambiguation.yaml --seed 0
python -m src.visualize --summary results/overall_summary.csv
python -m src.visualize_b11 --summary results/b11_flow_checkpoint_hardening_summary.csv
python -m src.visualize_b2 --summary results/b2_delayed_checkpoint_summary.csv
python -m src.visualize_b21 --summary results/b21_trace_hardening_summary.csv
python -m src.visualize_b21a --summary results/b21a_degeneracy_audit_summary.csv
python -m src.visualize_b22 --summary results/b22_selector_disentanglement_summary.csv
python -m src.visualize_b23 --summary results/b23_private_selector_summary.csv
python -m src.visualize_b3 --summary results/b3_active_inspection_summary.csv
python -m src.visualize_b31 --summary results/b31_inspection_audit_summary.csv
python -m src.visualize_b32 --summary results/b32_mechanism_inspection_summary.csv
python -m src.visualize_b4 --summary results/b4_intervention_summary.csv
python -m src.visualize_b41 --summary results/b41_intervention_audit_summary.csv
python -m src.visualize_b42 --summary results/b42_action_type_summary.csv
```

Run one model:

```powershell
python -m src.run_experiment --config configs/base.yaml --model schema_model --seed 0
```

## Expected Outputs

```text
results/behavior_summary.csv
results/structure_intervention_summary.csv
results/ood_summary.csv
results/overall_summary.csv
results/records.csv
results/null_control_summary.csv
results/flow_checkpoint_hardening_summary.csv
figures/*.png
reports/B_LINE_RESEARCH_PROGRAM.md
reports/B_LINE_PLOS_REPORT.md
reports/B_LINE_SELF_AUDIT.md
reports/B_LINE_SUBSTRATE_AUDIT.md
reports/B_LINE_SUBSTRATE_SEARCH.md
reports/B_LINE_FLOW_CHECKPOINT_HARDENING.md
reports/B_LINE_EVIDENCE_LADDER.md
results/substrate_audit.csv
results/b11_flow_checkpoint_hardening_summary.csv
results/b11_flow_checkpoint_records.csv
figures/b11_flow_checkpoint_hardening.png
figures/b11_attack_breakdown.png
reports/B1_1_FLOW_CHECKPOINT_HARDENING.md
reports/B1_1_FLOW_CHECKPOINT_SELF_AUDIT.md
results/b2_delayed_checkpoint_summary.csv
results/b2_delayed_checkpoint_records.csv
figures/b2_delayed_checkpoint_scores.png
figures/b2_delay_gate_breakdown.png
figures/b2_trace_intervention_effects.png
reports/B2_DELAYED_CHECKPOINT_REPORT.md
reports/B2_DELAYED_CHECKPOINT_SELF_AUDIT.md
results/b21_trace_hardening_summary.csv
results/b21_trace_hardening_records.csv
figures/b21_trace_hardening_scores.png
figures/b21_trace_attack_breakdown.png
figures/b21_trace_deletion_specificity.png
figures/b21_trace_compression_curve.png
reports/B2_1_TRACE_HARDENING_REPORT.md
reports/B2_1_TRACE_HARDENING_SELF_AUDIT.md
results/b21a_degeneracy_audit_summary.csv
results/b21a_degeneracy_audit_records.csv
results/b21a_per_attack_breakdown.csv
results/b21a_per_seed_breakdown.csv
results/b21a_predicted_region_distribution.csv
results/b21a_leakage_audit.csv
results/b21a_intervention_applicability.csv
results/b21a_baseline_comparison.csv
figures/b21a_score_degeneracy.png
figures/b21a_per_attack_breakdown.png
figures/b21a_predicted_region_distribution.png
figures/b21a_baseline_comparison.png
figures/b21a_ablation_effects.png
reports/B2_1A_TRACE_DEGENERACY_AUDIT.md
reports/B2_1A_TRACE_DEGENERACY_SELF_AUDIT.md
results/b22_selector_disentanglement_summary.csv
results/b22_selector_disentanglement_records.csv
results/b22_trace_provenance.csv
results/b22_selector_free_comparison.csv
results/b22_disagreement_episodes.csv
results/b22_source_specific_ablation.csv
results/b22_trace_scorer_correlation.csv
results/b22_shared_selector_ablation.csv
figures/b22_selector_usage.png
figures/b22_prediction_overlap.png
figures/b22_disagreement_divergence.png
figures/b22_ablation_effects.png
figures/b22_trace_scorer_correlation.png
reports/B2_2_TRACE_SELECTOR_DISENTANGLEMENT.md
reports/B2_2_TRACE_SELECTOR_SELF_AUDIT.md
results/b23_private_selector_summary.csv
results/b23_private_selector_records.csv
results/b23_selector_provenance.csv
results/b23_private_scorer_outputs.csv
results/b23_disagreement_results.csv
results/b23_source_ablation.csv
results/b23_regression_matrix.csv
figures/b23_private_selector_scores.png
figures/b23_provenance_breakdown.png
figures/b23_prediction_overlap.png
figures/b23_disagreement_divergence.png
figures/b23_source_ablation.png
figures/b23_regression_matrix.png
reports/B2_3_PRIVATE_TRACE_SELECTOR_REPORT.md
reports/B2_3_PRIVATE_TRACE_SELECTOR_SELF_AUDIT.md
results/b3_active_inspection_summary.csv
results/b3_active_inspection_records.csv
results/b3_baseline_comparison.csv
results/b3_trace_ablation_results.csv
results/b3_information_gain_records.csv
figures/b3_active_inspection_scores.png
figures/b3_information_gain.png
figures/b3_trace_vs_saliency_conflict.png
figures/b3_trace_ablation_effects.png
figures/b3_baseline_comparison.png
reports/B3_DELAYED_TRACE_GUIDED_ACTIVE_INSPECTION.md
reports/B3_ACTIVE_INSPECTION_SELF_AUDIT.md
results/b31_inspection_audit_summary.csv
results/b31_inspection_audit_records.csv
results/b31_inspect_region_overlap.csv
results/b31_inspection_provenance.csv
results/b31_inspection_scorer_correlation.csv
results/b31_disagreement_inspection_results.csv
results/b31_shared_policy_ablation.csv
results/b31_baseline_sanity.csv
results/b31_trace_ablation_specificity.csv
figures/b31_inspection_audit_scores.png
figures/b31_inspect_region_overlap.png
figures/b31_policy_provenance.png
figures/b31_scorer_correlation.png
figures/b31_disagreement_inspection.png
figures/b31_shared_policy_ablation.png
figures/b31_baseline_sanity.png
reports/B3_1_ACTIVE_INSPECTION_DEGENERACY_AUDIT.md
reports/B3_1_ACTIVE_INSPECTION_DEGENERACY_SELF_AUDIT.md
results/b32_mechanism_inspection_summary.csv
results/b32_mechanism_inspection_records.csv
results/b32_inspection_value_decomposition.csv
results/b32_goal_conditioned_results.csv
results/b32_mechanism_disagreement_results.csv
results/b32_family_ablation_results.csv
results/b32_baseline_comparison.csv
figures/b32_mechanism_inspection_scores.png
figures/b32_family_specific_targets.png
figures/b32_value_decomposition.png
figures/b32_goal_conditioned_switching.png
figures/b32_mechanism_disagreement.png
figures/b32_family_ablation.png
reports/B3_2_MECHANISM_DISAMBIGUATING_ACTIVE_INSPECTION.md
reports/B3_2_MECHANISM_DISAMBIGUATING_SELF_AUDIT.md
results/b4_intervention_summary.csv
results/b4_intervention_records.csv
results/b4_baseline_comparison.csv
results/b4_trace_ablation_results.csv
results/b4_wrong_region_penalty.csv
results/b4_family_intervention_results.csv
figures/b4_intervention_scores.png
figures/b4_action_type_accuracy.png
figures/b4_wrong_region_penalty.png
figures/b4_trace_ablation_effects.png
figures/b4_family_intervention_targets.png
figures/b4_baseline_comparison.png
reports/B4_DELAYED_TRACE_GUIDED_INTERVENTION.md
reports/B4_INTERVENTION_SELF_AUDIT.md
results/b41_intervention_audit_summary.csv
results/b41_intervention_audit_records.csv
results/b41_action_overlap.csv
results/b41_action_provenance.csv
results/b41_action_scorer_correlation.csv
results/b41_shared_action_policy_ablation.csv
results/b41_wrong_action_stress.csv
results/b41_baseline_sanity.csv
results/b41_trace_ablation_specificity.csv
figures/b41_intervention_audit_scores.png
figures/b41_action_overlap.png
figures/b41_action_type_distribution.png
figures/b41_action_policy_provenance.png
figures/b41_action_scorer_correlation.png
figures/b41_shared_policy_ablation.png
figures/b41_wrong_action_stress.png
figures/b41_baseline_sanity.png
figures/b41_trace_ablation_specificity.png
reports/B4_1_INTERVENTION_DEGENERACY_AUDIT.md
reports/B4_1_INTERVENTION_DEGENERACY_SELF_AUDIT.md
results/b42_action_type_summary.csv
results/b42_action_type_records.csv
results/b42_action_type_value_table.csv
results/b42_correct_region_wrong_action.csv
results/b42_family_action_mapping.csv
results/b42_action_type_counterfactual.csv
results/b42_fixed_action_baseline.csv
results/b42_action_type_ablation.csv
results/b42_action_type_ood.csv
figures/b42_action_type_scores.png
figures/b42_action_type_distribution.png
figures/b42_correct_region_wrong_action.png
figures/b42_family_action_mapping.png
figures/b42_counterfactual_sensitivity.png
figures/b42_fixed_action_baseline.png
figures/b42_action_type_ood.png
reports/B4_2_ACTION_TYPE_DISAMBIGUATION.md
reports/B4_2_ACTION_TYPE_DISAMBIGUATION_SELF_AUDIT.md
```

## B1.1 Reviewer Hardening

B1.1 targets only `flow_checkpoint_model`, the current PLOS v1 candidate. It
does not change PLOS v1 scoring. It adds six reviewer attacks:

- dynamic decoy checkpoint
- delayed checkpoint
- competing checkpoints
- checkpoint relocation OOD
- causal deletion vs visual deletion
- anti-prior world

If B1.1 passes, the flow-checkpoint substrate remains a high-prior but stronger
PLOS foothold. If it fails, the prior PLOS pass is likely checkpoint-prior,
saliency, or weak causal-intervention dependent.

Passing B1.1 does not prove blank-slate emergence, general physical reasoning,
real-world cognition, or language-free intelligence.

## B2 Delayed Checkpoints

B2 does not repair B1.1. It targets the gap B1.1 exposed:
`flow_checkpoint_model` is useful but short-horizon, and fails delayed
checkpoint selection. B2 compares that baseline with:

- `recurrent_flow_checkpoint_model`
- `field_memory_model`
- `schema_memory_model`

The test asks whether delayed operational checkpoints can be carried by
temporal memory, field trace, or sparse schema memory under behavior, causal
trace intervention, and heldout-delay OOD gates.

A passing B2 model is a delayed PLOS candidate in this 64x64 toy world. It is
not proof of blank-slate emergence, physics understanding, or general
pre-linguistic intelligence.

See `reports/B_LINE_EVIDENCE_LADDER.md` for the current B-line evidence ladder
from PLOS v1 through B1.1, B2, B2.1, B2.1a, B2.2, B2.3, B3, B3.1, B3.2, B4, B4.1, and B4.2.

## B2.1 Trace Hardening

B2.1 attacks the trace-bearing models from B2. It asks whether temporal memory,
field trace, and sparse schema memory are actually used as causal delayed
trace, rather than acting as another structure prior or clean-generator cue.

The attacks are false delayed trace, trace swap, trace deletion specificity,
multi-source trace conflict, noisy trace, trace length extrapolation, and trace
compression pressure.

Passing B2.1 means only that a model remains a hardened trace-bearing substrate
candidate in the toy PLOS environment.

## B2.1a Degeneracy Audit

B2.1a audits why all three trace-bearing models received the same B2.1 score.
It checks per-attack and per-seed breakdowns, predicted-region distributions,
ground-truth leakage, intervention applicability, random/oracle baselines, and
trace/no-trace ablations.

The current B2.1a result flags exact cross-model prediction matching and
identical per-attack metrics. This means B2.1 is not yet reliable evidence for
independent trace mechanisms, even though leakage, random passability,
unsupported intervention, and missing trace-dependency risks are reduced.

## B2.2 Trace Selector Disentanglement

B2.2 asks whether B2/B2.1 success comes from independent recurrent, field, and
schema trace mechanisms, or from a shared trace selector path.

The current B2.2 result is deliberately conservative: all three base
trace-bearing models report `shared_selector_usage_rate=1.000` and
`cross_model_exact_prediction_match_rate=1.000`, so
`b22_disentanglement_score=0.000`.

Interpretation: B2/B2.1 should currently be treated as evidence that a
trace-bearing path is useful, but not as evidence that recurrent, field, and
schema trace mechanisms are independently separated.

## B2.3 Private Trace Selector Construction

B2.3 reconstructs `recurrent_flow_checkpoint_model`, `field_memory_model`, and
`schema_memory_model` so their delayed checkpoint selection comes from
model-private trace scorers rather than `delayed_common.select_delayed_region`.

The current B2.3 result reports:

```text
shared_selector_usage_rate = 0.000
cross_model_exact_prediction_match_rate = 0.000
disagreement_episode_divergence = 1.000
b23_private_selector_score = 0.972
```

Interpretation: B2.3 reduces the shared-selector explanation and supports
partial mechanism separation under the current toy diagnostics. It still does
not prove blank-slate emergence, complete mechanism independence, or general
delayed causality.

## B3 Active Inspection

B3 keeps the B2.3 private trace selectors fixed and asks whether delayed trace
can guide budgeted active inspection. The model gets one 8x8 region inspect
action and must prefer delayed information value over visual saliency or
short-horizon checkpoint cues.

The current B3 result reports:

```text
trace_guided_inspection_accuracy = 1.000
trace_vs_saliency_rejection = 1.000
delayed_information_gain = 1.000
trace_ablation_inspection_drop = 1.000
delay_ood_inspection_accuracy = 0.975
b3_active_inspection_score = 0.985
```

Interpretation: in the toy PLOS environment, B2.3 private delayed traces can
guide budgeted active inspection under the current gates. This does not prove
general active intelligence, real-world inspection ability, human-like
attention, or language-free cognition.

## B3.1 Active Inspection Degeneracy Audit

B3.1 audits the identical B3 scores. It checks whether the three private-trace
models choose the same inspect region per episode, whether a shared inspection
policy is used, whether private trace inspection scorers are active, and
whether disagreement-inspection episodes force divergence.

The current B3.1 result reports:

```text
cross_model_inspect_region_match_rate = 1.000
shared_inspection_policy_usage_rate = 0.000
private_trace_inspection_score_usage_rate = 1.000
inspection_scorer_specificity = 1.000
disagreement_inspection_divergence = 1.000
private_trace_over_non_trace_ratio = 1000000.000
b31_inspection_audit_score = 0.000
```

Interpretation: B3 remains evidence that the shared trace-guided inspection path
is useful, but B3.1 does not support independent recurrent / field / schema
active-inspection mechanism validation. The failing gate is ordinary B3
per-episode inspect-region overlap: all three models pick the same region on
the standard B3 episodes.

## B3.2 Mechanism-Disambiguating Active Inspection

B3.2 addresses the B3.1 same-region degeneracy by constructing episodes with
different recurrent-, field-, and schema-optimal inspect regions. It adds
multi-objective inspection values, a non-linguistic goal code, mechanism
disagreement episodes, and family-specific trace ablation.

The current B3.2 result reports:

```text
family_specific_inspection_accuracy = 1.000
task_conditioned_switch_accuracy = 1.000
mechanism_disagreement_rate = 1.000
cross_model_same_region_rate = 0.000
family_specific_trace_ablation_drop = 1.000
non_target_family_stability = 1.000
b32_mechanism_inspection_score = 0.963
```

Interpretation: B3.2 reduces the B3.1 same-region degeneracy under the current
toy diagnostics and supports partial mechanism-disambiguated active inspection
behavior. It does not prove real active intelligence, real-world inspection
ability, complete mechanism independence, or language-free cognition solved.

## B4 Delayed Trace-Guided Intervention / Action Selection

B4 moves from inspection to a minimal local intervention/action diagnostic. It
keeps the 64x64 toy world and a constrained action budget of one, asking whether
private delayed traces can select both where to intervene and which discrete
local action type to apply.

The current B4 result reports:

```text
trace_guided_intervention_accuracy = 1.000
intervention_region_accuracy = 1.000
action_type_accuracy = 1.000
outcome_improvement = 1.000
intervention_vs_inspection_gain = 1.000
wrong_region_penalty_sensitivity = 0.900
family_specific_intervention_accuracy = 1.000
trace_ablation_intervention_drop = 1.000
delay_ood_intervention_accuracy = 1.000
b4_intervention_score = 0.985
```

Interpretation: in the toy PLOS environment, B3.2 private delayed traces can
guide minimal local intervention/action selection under the current diagnostic
gates. This does not prove real control, robotics deployment, real engineering
intervention, general active intelligence, or language-free cognition solved.

## B4.1 Intervention Degeneracy Audit

B4.1 audits whether the high B4 intervention score is better explained by
private trace-guided action selection or by shared action policy, fixed action
type, value leakage, weak baselines, or nonspecific trace ablation.

The current B4.1 result reports:

```text
cross_model_exact_action_match_rate = 0.000
exact_all_model_same_action_rate = 0.000
fixed_action_type_rate = 1.000
shared_action_policy_usage_rate = 0.000
private_trace_action_score_usage_rate = 1.000
action_scorer_specificity = 0.945
wrong_action_penalty = 0.650
wrong_region_penalty = 0.900
private_trace_ablation_drop = 1.000
action_type_shift_after_trace_ablation = 0.000
value_leakage_count = 0
b41_intervention_audit_score = 0.000
```

Interpretation: B4.1 does not erase B4's trace-guided intervention-path result,
but it blocks an independent recurrent / field / schema intervention-mechanism
claim. The current bottleneck is fixed action-type selection: each model uses a
family-fixed action type, and trace ablation changes region choice but not action
type.

## B4.2 Action-Type Disambiguation

B4.2 directly attacks the fixed action-type shortcut found by B4.1. It creates
action-type-specific intervention targets where correct region plus wrong action
is penalized, adds fixed-action and random-action-type baselines, and tests OOD
family-action mappings.

The current B4.2 result reports:

```text
fixed_action_type_rate = 0.500
action_type_accuracy = 1.000
region_accuracy = 1.000
joint_region_action_accuracy = 1.000
correct_region_wrong_action_penalty = 0.900
action_type_counterfactual_sensitivity = 0.900
family_action_diversity = 0.500
action_type_shift_after_trace_ablation = 0.500 / 1.000
action_type_ablation_drop = 0.500 / 1.000
action_type_ood_accuracy = 1.000
b42_action_type_score = 0.877 / 0.937
```

Interpretation: B4.2 reduces the B4.1 fixed-action-type shortcut under the
current toy diagnostics. Private delayed traces now support intervention-region
selection and differentiated action-type selection in action-type-specific
episodes. This still does not prove real control, robotics ability, engineering
deployment, or general active intelligence.

## Boundary

Supported only if results justify it:

```text
In a minimal 2D continuous world, some substrates show evidence of
pre-linguistic operational structure when they pass behavior, structural
intervention, and OOD gates.
```

Unsupported:

```text
general intelligence
LLM understanding
real-world physical reasoning
deployment-ready robot cognition
proof that objects/events are universal structures
proof that language is irrelevant
```
