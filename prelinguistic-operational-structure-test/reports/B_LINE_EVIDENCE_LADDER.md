# B-Line Evidence Ladder

## From PLOS v1 to B3

This document summarizes the current evidence chain for the B-line:

```text
W -> O1 -> O2
```

where:

- `W` = continuous world process / fact-occurrence layer
- `O1` = pre-linguistic trajectory schema
- `O2` = operational structure for action, intervention, inspection, and delayed relevance

The B-line does **not** test language-level relation audit.  
It tests whether operational structure can be supported by non-linguistic continuous dynamics under behavior, intervention, and OOD gates.

---

## 0. Executive Summary

The B-line has now produced an evidence ladder with delayed-trace hardening and selector-provenance audit:

```text
PLOS v1:
  Establishes a minimal diagnostic program for pre-linguistic operational structure.

B1.1:
  Attacks the only PLOS v1 candidate, flow_checkpoint_model.

B2:
  Tests whether delayed operational checkpoints require trace-bearing substrates.

B2.1/B2.1a:
  Hardens trace use, then detects identical prediction degeneracy.

B2.2:
  Tests whether the trace selector is independently separated across substrates.

B2.3:
  Reconstructs the trace-bearing models with private trace selectors.

B3:
  Tests whether private delayed trace guides budgeted active inspection.
```

Current best interpretation:

> PLOS v1 found a high-prior short-horizon checkpoint candidate.  
> B1.1 showed that this candidate is not merely visual saliency, but it fails delayed causal checkpoint selection.  
> B2 supports the diagnosis that delayed operational checkpoints require trace-bearing paths.
> B2.1a and B2.2 show that the current recurrent/field/schema models should not yet be treated as independent trace mechanisms because they share effectively identical selector behavior.
> B2.3 reduces that shared-selector interpretation by replacing the shared selector with private recurrent, field, and schema trace scorers.
> B3 shows that those private traces can guide a one-region active inspection decision under toy budgeted-inspect gates.

Current strongest claim:

> In the toy PLOS world, short-horizon checkpoint structure is insufficient for delayed operational relevance. Delayed operational structure appears to require a trace-bearing path. After B2.3, the current private selectors show partial mechanism separation under toy diagnostics. B3 adds evidence that private delayed trace can guide budgeted active inspection, while still not proving natural emergence, complete independence, or general active intelligence.

Current unsupported claims:

- The system has general physical intelligence.
- Operational structure has emerged from a blank-slate substrate.
- The W -> O1 -> O2 framework is proven.
- The model understands physics in a human-like way.
- The result transfers to real-world robotics, construction, or geotechnical monitoring.
- The system has general active inspection or control intelligence.

---

## 1. Stage 1: PLOS v1

### 1.1 Question

PLOS v1 asks:

> Can a model form pre-linguistic operational structure from continuous 2D dynamics without language labels?

The evidence rule is:

```text
Behavior + Structural Intervention + OOD
```

Behavior alone is not sufficient.

Prediction accuracy, tracking accuracy, event accuracy, or inspection accuracy alone cannot count as evidence for operational structure.

---

### 1.2 Tested Model Families

PLOS v1 includes the following types of systems:

| Model / branch | Role |
|---|---|
| `pixel_predictor` | First-order pixel prediction baseline |
| `trajectory_memory` | Trajectory-fragment memory baseline |
| `world_model` | Strong first-order latent sequence baseline |
| `slot_model` | Object-biased operational-structure candidate |
| `field_model` | Non-object-centric field-form candidate |
| `schema_model` | Field-first hybrid schema candidate |
| `predictive_coding_model` | Prediction-error substrate |
| `patch_graph_model` | Local patch dynamics substrate |
| `koopman_model` | Low-rank dynamics substrate |
| `flow_checkpoint_model` | Flow/checkpoint operational substrate |

---

### 1.3 Result

PLOS v1 found one qualified candidate:

```text
flow_checkpoint_model
```

Reported PLOS v1 result:

```text
flow_checkpoint_model:
  behavior_score = 0.910
  structure_intervention_score = 0.262
  ood_score = 0.903
  plos_candidate_score = 0.434
```

Other tested substrates did not pass the full PLOS candidate gate.

---

### 1.4 Supported Claim

PLOS v1 supports this limited claim:

> A high-prior checkpoint substrate can pass the first PLOS evidence rule in a minimal 2D world.

More specifically:

> Checkpoint-like operational structures are usable and testable under behavior, structural intervention, and OOD gates.

---

### 1.5 Unsupported Claim

PLOS v1 does **not** support:

```text
flow_checkpoint_model discovered operational structure from scratch
checkpoint structure is blank-slate emergence
object/event/checkpoint structures are universal
general physical reasoning has been achieved
language-free intelligence has been solved
```

---

### 1.6 False-Positive Exclusions

PLOS v1 excludes several weak interpretations:

| False positive | How PLOS v1 addresses it |
|---|---|
| Prediction accuracy alone | Requires structural intervention and OOD gates |
| Trajectory memory | Includes trajectory-memory baseline and OOD tests |
| Object-centric bias | Includes field-based and non-object-centric baselines |
| Language-level explanation | No language labels are given in B-line v1 |
| Pure static structure prior | Uses null/static controls and substrate audit |
| Generic latent prediction | Requires model-specific structural intervention |

---

### 1.7 Remaining Weakness

The only candidate, `flow_checkpoint_model`, has a strong architectural prior:

```text
continuity checkpoint
occlusion checkpoint
predicted collision checkpoint
force anomaly checkpoint
motion midpoint checkpoint
```

Therefore, PLOS v1 is best interpreted as:

> A checkpoint-prior foothold, not a proof of spontaneous operational-structure emergence.

---

## 2. Stage 2: B1.1 Flow-Checkpoint Reviewer Hardening

### 2.1 Question

B1.1 asks:

> Is the PLOS v1 candidate robust, or is it merely exploiting checkpoint priors, visual saliency, or weak intervention metrics?

B1.1 targets only:

```text
flow_checkpoint_model
```

It does not modify PLOS v1 scoring.

---

### 2.2 Attack Set

B1.1 adds six reviewer attacks:

| Attack | Purpose |
|---|---|
| Dynamic decoy checkpoint | Tests whether moving visual decoys attract checkpoint selection |
| Delayed checkpoint | Tests whether future-critical but currently non-salient checkpoints are selected |
| Competing checkpoints | Tests whether the model chooses operational value rather than fixed checkpoint priority |
| Checkpoint relocation OOD | Tests checkpoint generalization to unusual spatial positions |
| Causal deletion vs visual deletion | Tests whether causal regions matter more than visual-salient noncausal regions |
| Anti-prior world | Tests whether checkpoint priors fail in deliberately misleading worlds |

---

### 2.3 Result

B1.1 result:

```text
candidate_gate_preserved = 1.000
dynamic_decoy_rejection = 0.843
delayed_checkpoint_accuracy = 0.000
competing_checkpoint_choice = 0.970
relocation_ood_stability = 0.840
causal_over_visual_deletion_ratio = 2.166
anti_prior_survival = 0.850
causal_endpoint_shift = 0.375
b11_hardening_score = 0.000
```

The model passes several important reviewer attacks, but fails the delayed checkpoint gate.

---

### 2.4 Supported Claim

B1.1 supports this refined claim:

> `flow_checkpoint_model` is not a trivial visual-saliency model. It is robust to dynamic decoys, competing checkpoints, relocation OOD, causal-vs-visual deletion, and anti-prior traps under the current toy setup.

---

### 2.5 Critical Failure

However, B1.1 falsifies the strong interpretation of `flow_checkpoint_model`.

It fails:

```text
delayed_checkpoint_accuracy = 0.000
```

Therefore, the correct interpretation is:

> `flow_checkpoint_model` is a short-horizon checkpoint substrate, not a delayed causal operational-structure substrate.

---

### 2.6 Unsupported Claim

B1.1 does **not** support:

```text
flow_checkpoint_model handles delayed operational relevance
flow_checkpoint_model has robust causal trace memory
flow_checkpoint_model is sufficient for long-horizon operational structure
```

---

### 2.7 False-Positive Exclusions

B1.1 excludes additional false positives:

| False positive | Status after B1.1 |
|---|---|
| Moving decoy saliency | Mostly excluded by dynamic decoy rejection |
| Fixed checkpoint priority | Mostly excluded by competing checkpoint test |
| Common spatial location memorization | Partly excluded by relocation OOD |
| Visual saliency mistaken for causal relevance | Partly excluded by causal-vs-visual deletion |
| Simple checkpoint-prior trap | Partly excluded by anti-prior world |
| Delayed causal checkpoint | **Not excluded; model fails here** |

---

### 2.8 Main Diagnostic Value

B1.1 localizes the main failure:

```text
short-horizon checkpoint works
delayed causal checkpoint fails
```

This becomes the transition point to B2.

---

## 3. Stage 3: B2 Delayed Operational Checkpoint Substrate

### 3.1 Question

B2 asks:

> What kind of substrate is needed to support delayed operational checkpoints?

B2 does not try to repair `flow_checkpoint_model` directly.

Instead, it tests whether delayed operational relevance requires trace-bearing substrates.

---

### 3.2 Motivation

B1.1 showed:

```text
flow_checkpoint_model:
  robust under many local attacks
  but delayed_checkpoint_accuracy = 0.000
```

This suggests that short-horizon checkpoint structure is not enough.

Real physical systems often involve delayed relevance:

```text
rainfall now
pore pressure later
displacement later
crack development later
risk exposure last
```

Therefore, B2 asks whether delayed checkpoints require:

```text
temporal memory
field trace
sparse schema memory
```

---

### 3.3 Tested Models

B2 adds:

| Model | Role |
|---|---|
| `flow_checkpoint_model` | Short-horizon checkpoint baseline |
| `recurrent_flow_checkpoint_model` | Adds recurrent temporal memory to checkpoint selection |
| `field_memory_model` | Uses field-trace / delayed influence fields |
| `schema_memory_model` | Uses sparse schema memory slots for delayed candidates |

---

### 3.4 B2 Tasks

B2 includes:

| Task | Purpose |
|---|---|
| Delayed checkpoint | Select future-critical but currently non-salient checkpoint |
| Multi-delay checkpoint | Handle multiple candidate delays |
| Early-saliency rejection | Avoid selecting immediate but noncausal salient region |
| Delay OOD | Generalize to heldout delays |
| Causal trace intervention | Verify that delayed trace is behaviorally causal |
| Non-trace control | Ensure trace intervention is specific |

---

### 3.5 Result

B2 numerical table inserted from:

```text
results/b2_delayed_checkpoint_summary.csv
```

Current result after running B2:

> B2 supports the B1.1 diagnosis: the short-horizon checkpoint substrate is insufficient. Delayed checkpoint performance requires additional trace-bearing substrates such as temporal memory, field trace, or schema memory.

Suggested table format:

| Model | Delayed accuracy | Multi-delay | Early-saliency rejection | Delay OOD | Trace intervention drop | B2 score | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `flow_checkpoint_model` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | Short-horizon baseline fails delayed gates |
| `recurrent_flow_checkpoint_model` | 1.000 | 1.000 | 1.000 | 1.000 | 0.954 | 1.000 | Temporal-memory candidate passes B2 |
| `field_memory_model` | 1.000 | 1.000 | 1.000 | 1.000 | 0.954 | 1.000 | Field-trace candidate passes B2 |
| `schema_memory_model` | 1.000 | 1.000 | 1.000 | 1.000 | 0.954 | 1.000 | Sparse-schema-memory candidate passes B2 |

---

### 3.6 Supported Claim

B2 supports this claim, assuming the reported run shows trace-bearing models outperform the short-horizon baseline under B2 gates:

> Delayed operational checkpoints require a trace-bearing substrate. Short-horizon checkpoint selection is insufficient when operational relevance is delayed.

More carefully:

> In the current toy PLOS environment, delayed checkpoint performance improves only when the substrate can preserve a temporal, field, or schema trace of future-relevant regions.

---

### 3.7 Unsupported Claim

B2 does **not** support:

```text
delayed operational structure emerged from a blank slate
trace-bearing substrates are universal requirements
human-like trajectory schema has been achieved
the model understands delayed causality generally
the result transfers to real geotechnical monitoring
```

---

### 3.8 False-Positive Exclusions

B2 adds a new layer of false-positive control:

| False positive | B2 check |
|---|---|
| Short-horizon checkpoint saliency | Delayed checkpoint task |
| Immediate visual saliency | Early-saliency rejection |
| Single fixed delay | Multi-delay and heldout-delay OOD |
| Trace not actually used | Causal trace intervention |
| General hidden-state fragility | Non-trace stability control |
| Hard-coded delayed rule | Compare multiple substrate types and require intervention evidence |

---

## 4. Evidence Ladder Summary

The B-line evidence ladder is now:

```text
Level 0:
  First-order prediction and memory baselines.
  Result:
    behavior may be useful, but no sufficient evidence for O.

Level 1:
  PLOS v1 flow_checkpoint_model.
  Result:
    short-horizon checkpoint substrate passes initial behavior + intervention + OOD.

Level 2:
  B1.1 hardening.
  Result:
    flow_checkpoint is not trivial saliency, but fails delayed checkpoint.

Level 3:
  B2 delayed checkpoint.
  Result:
    delayed operational checkpoint requires trace-bearing substrate.
```

---

## 5. Updated Main Claim

The current strongest B-line claim is:

> Pre-linguistic operational structure is not exhausted by visible checkpoint selection. When operational relevance is delayed, the substrate must preserve a causal trace across time. In the current PLOS toy world, delayed checkpoint performance requires trace-bearing substrates such as recurrent memory, field memory, or sparse schema memory.

Shorter version:

> Short-horizon checkpoint structure is not enough; delayed operational relevance requires trace-bearing structure.

---

## 6. Updated Boundary

Supported:

```text
PLOS v1 defines a minimal behavior + intervention + OOD diagnostic.
flow_checkpoint_model is a useful short-horizon checkpoint foothold.
B1.1 shows the short-horizon candidate fails delayed checkpoint.
B2 suggests delayed checkpoint requires trace-bearing substrate.
```

Not supported:

```text
blank-slate operational-structure emergence
general physical intelligence
human-like pre-linguistic cognition
LLM understanding
real-world deployment
universal object/event/checkpoint theory
```

---

## 7. B2.1 Trace Hardening

B2 introduces trace-bearing substrates.  
But this creates a new false-positive risk:

> A trace-bearing model may pass because trace structure is injected as architecture, not because it learns to use delayed operational relevance robustly.

B2.1 hardens the trace-bearing result with false-trace, swap, deletion-specificity, conflict, noise, extrapolation, and compression attacks.

---

## 8. B2.1a Degeneracy Audit

B2.1 produced identical scores for all three trace-bearing models:

```text
recurrent_flow_checkpoint_model = 0.959
field_memory_model = 0.959
schema_memory_model = 0.959
```

B2.1a audits this score degeneracy.

Key result:

```text
score_degeneracy_detected = 1.000
all_attacks_identical_flag = 1.000
cross_model_exact_prediction_match_rate = 1.000
leakage_count = 0.000
random_b21_score = 0.000
oracle_b21_score = 1.000
trace_family_ablation_drop = 1.000
no_trace_ablation_drop = 1.000
b21a_degeneracy_audit_score = 0.000
```

Interpretation:

> B2.1a does not find ground-truth key leakage, random baseline passability, unsupported interventions, or missing trace dependency. However, it does find exact cross-model prediction matching and identical per-attack metrics. Therefore B2.1 is not yet reliable evidence for independent trace-bearing mechanisms.

This does not erase B2's diagnostic value. It narrows the next problem:

```text
trace-bearing behavior exists
but independent substrate mechanisms are not yet separated
```

---

## 9. B2.2 Trace Selector Disentanglement

B2.2 directly tests the next narrowed question:

```text
Are recurrent, field, and schema trace-bearing models using independent trace
selection mechanisms, or are they all routing through the same selector?
```

Key result:

```text
shared_selector_usage_rate = 1.000
cross_model_exact_prediction_match_rate = 1.000
disagreement_episode_divergence = 0.023
b22_disentanglement_score = 0.000
```

Interpretation:

> B2.2 shows that B2/B2.1 success should currently be interpreted as shared trace-selector success, not as independent recurrent/field/schema trace mechanism validation.

This is not a failure of the whole B-line. It sharpens the next design target:

```text
trace-bearing path is useful
but trace selector provenance is not yet disentangled
```

---

## 10. B2.3 Private Trace Selector Construction

B2.3 reconstructs the three trace-bearing models so that delayed checkpoint
selection is performed by model-private scorers:

```text
recurrent_flow_checkpoint_model -> recurrent_memory private scorer
field_memory_model              -> field_trace private scorer
schema_memory_model             -> schema_memory private scorer
```

Key result:

```text
shared_selector_usage_rate = 0.000
model_private_score_usage_rate = 1.000
cross_model_exact_prediction_match_rate = 0.000
disagreement_episode_divergence = 1.000
b2_delayed_score = 1.000
b21_trace_hardening_score = 0.960
b23_private_selector_score = 0.972
```

Interpretation:

> B2.3 reduces the shared-selector explanation of B2/B2.1 by replacing the shared selector with private trace scorers. Under the current toy diagnostics, recurrent, field, and schema trace paths show partial mechanism separation.

Boundary:

```text
B2.3 does not prove blank-slate emergence.
B2.3 does not prove complete mechanism independence.
B2.3 does not prove general delayed causality.
```

---

## 11. B3 Delayed Trace-Guided Active Inspection

B3 keeps the B2.3 private trace selectors fixed and tests whether they can
guide a budgeted inspect decision:

```text
private delayed trace
  -> one 8x8 inspect-region choice
  -> delayed information gain
```

The key conflict is:

```text
visual saliency / short-horizon checkpoint
  vs
currently non-salient but delayed-information-value trace region
```

Key result:

```text
trace_guided_inspection_accuracy = 1.000
trace_vs_saliency_rejection = 1.000
delayed_information_gain = 1.000
inspection_value_gain_over_random = 0.936
inspection_value_gain_over_saliency = 0.950
inspection_value_gain_over_short_horizon = 0.950
trace_ablation_inspection_drop = 1.000
delay_ood_inspection_accuracy = 0.975
oracle_inspection_score = 1.000
random_inspection_score = 0.006
b3_active_inspection_score = 0.985
```

Interpretation:

> B3 shows that, in the toy PLOS environment, B2.3 private delayed traces can guide budgeted active inspection. The models choose delayed trace regions over saliency and short-horizon baselines, achieve delayed information gain, and lose inspection performance after private trace ablation.

Boundary:

```text
B3 does not prove general active intelligence.
B3 does not prove real-world robot inspection.
B3 does not prove human-like attention.
B3 does not prove language-free cognition solved.
```

---

## 12. Recommended Repository Placement

Save this document as:

```text
prelinguistic-operational-structure-test/reports/B_LINE_EVIDENCE_LADDER.md
```

Also update README with a short pointer:

```text
See reports/B_LINE_EVIDENCE_LADDER.md for the current B-line evidence ladder from PLOS v1 through B1.1, B2, B2.1, B2.1a, B2.2, B2.3, and B3.
```

---

## 13. One-Sentence Project Status

> B-line has progressed from diagnosing short-horizon operational checkpoints to identifying delayed causal trace, exposing shared-selector dependence, constructing private trace selectors with partial mechanism separation, and showing that those private traces can guide budgeted active inspection in the toy PLOS environment.
