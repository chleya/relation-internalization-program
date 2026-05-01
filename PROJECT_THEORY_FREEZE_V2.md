# PROJECT_THEORY_FREEZE_V2.md

**Project:** Relation Internalization Program / Pre-Linguistic Operational Structure  
**Freeze version:** V2  
**Freeze date:** 2026-05-01  
**Status:** Updated after B4.2 Action-Type Disambiguation passed.  
**Purpose:** Update the project theory freeze after B4.2 resolved the fixed-action-type shortcut exposed by B4.1. Reassess current strongest claim, claim boundary, unresolved bottleneck, and next-stage direction.

---

## 0. V2 one-sentence freeze

The project has now advanced from relation internalization to pre-linguistic operational structure internalization. In the current toy PLOS environment, private delayed traces can support delayed checkpoint, private selector, active inspection, mechanism-disambiguated inspection, intervention-region selection, and differentiated action-type selection. The current bottleneck has shifted from action-type disambiguation to closed-loop operation: whether the system can inspect to reduce uncertainty, update its trace, intervene based on the update, observe consequences, and revise the trace.

---

## 1. Original question and its transformation

The project began with the question:

```text
Does the model internalize relations, or merely predict outputs?
```

The early A-line framed relation internalization as an internal relation representation:

```text
observations / support examples
→ relation representation R_hat
→ transfer
→ counterfactual
→ local edit
→ audit
→ budgeted inspect
```

This was useful but still too close to symbolic relation tables and language-covered descriptions.

The deeper correction was:

```text
facts happen first;
language symbols later cover them.
```

This forced the project to move from symbolic relation internalization to pre-linguistic operational structure.

---

## 2. W → O₁ → O₂ → L framework

The project’s current theoretical backbone is:

```text
W → O₁ → O₂ → L
```

### 2.1 W: continuous fact occurrence layer

W is the world process itself.

Examples:

```text
object rolls
object occludes
object collides
local field changes trajectory
delayed influence appears later
slope moves
crack expands
pore pressure rises
```

W is not language and not intelligence. It is the fact stream.

---

### 2.2 O₁: trajectory schema / delayed trace layer

O₁ is the first operational compression of W.

It includes:

```text
object persistence
motion continuity
occlusion continuation
crossing identity preservation
collision boundary
delayed trace
checkpoint
trace memory
```

O₁ is pre-linguistic. It does not need a relation label.

---

### 2.3 O₂: actionable operational structure layer

O₂ is O₁ made usable for action.

It includes:

```text
inspect target
intervention target
action region
action type
wrong-region penalty
wrong-action penalty
budgeted decision
action value
```

O₂ answers:

```text
where should I inspect?
where should I intervene?
what action type should I use?
what happens if the region is right but the action type is wrong?
```

---

### 2.4 L: language cover / audit layer

L describes O in language.

Examples:

```text
"the delayed trace region should be inspected"
"the action should stabilize the trace region"
"rainfall delayed pore pressure rise"
```

L is not the structure itself. It is an external cover, report, or audit interface.

A language system may produce L without robust O.

Therefore:

```text
L → L is insufficient.
W → O₁ → O₂ → L is the target.
```

---

## 3. Current evidence ladder after B4.2

This section freezes the current B-line evidence chain.

---

### 3.1 PLOS v1

**Question:**  
Can any model pass a minimal pre-linguistic operational structure test?

**Result:**  
`flow_checkpoint_model` became the initial short-horizon checkpoint candidate.

**Claim:**  
Short-horizon checkpoint structure can capture some local operational cues.

**Boundary:**  
No delayed trace, no active inspection, no intervention.

---

### 3.2 B1.1 Flow-Checkpoint Reviewer Hardening

**Question:**  
Is `flow_checkpoint_model` merely a saliency shortcut?

**Result:**  
It survived some local tests but failed delayed checkpoint.

**Claim:**  
Short-horizon checkpoint is not entirely trivial.

**Failure:**  
Delayed checkpoint accuracy collapsed.

**Interpretation:**  
Short-horizon checkpoint is insufficient for delayed operational structure.

---

### 3.3 B2 Delayed Operational Checkpoint Substrate

**Question:**  
What substrate handles delayed checkpoint?

**Added models:**

```text
recurrent_flow_checkpoint_model
field_memory_model
schema_memory_model
```

**Claim:**  
Trace-bearing paths can solve delayed checkpoint.

**Boundary:**  
Initial success did not yet prove independent mechanisms.

---

### 3.4 B2.1 Trace-Bearing Substrate Hardening

**Question:**  
Do trace-bearing paths survive trace attacks?

**Attacks included:**

```text
false delayed trace
trace swap
trace deletion specificity
multi-source conflict
noisy delayed trace
trace length extrapolation
compression pressure
```

**Result:**  
Trace-bearing models passed.

**Boundary:**  
Identical scores raised a degeneracy concern.

---

### 3.5 B2.1a Trace Hardening Score Degeneracy Audit

**Question:**  
Why did recurrent / field / schema models score identically?

**Result:**  
No ground-truth leakage, no random pass, no inactive interventions.  
But the models produced identical per-episode predictions.

**Interpretation:**  
B2/B2.1 supported trace-bearing path usefulness, not independent mechanism validation.

---

### 3.6 B2.2 Trace Selector Disentanglement

**Question:**  
Is the success caused by a shared trace selector?

**Result:**  
Conservative failure.

**Interpretation:**  
B2/B2.1 should be interpreted as shared trace-selector success.

---

### 3.7 B2.3 Private Trace Selector Construction

**Question:**  
Can recurrent / field / schema trace selectors be made private?

**Reported results:**

```text
shared_selector_usage_rate = 0.000
model_private_score_usage_rate = 1.000
cross_model_exact_prediction_match_rate = 0.000
disagreement_episode_divergence = 1.000
b2_delayed_score = 1.000
b21_trace_hardening_score = 0.960
b23_private_selector_score = 0.972
```

**Claim:**  
Private trace selectors reduce the shared-selector explanation and support partial mechanism separation under toy diagnostics.

**Boundary:**  
Still not blank-slate emergence or general delayed causality.

---

### 3.8 B3 Delayed Trace-Guided Active Inspection

**Question:**  
Can private delayed trace guide active inspection under budget?

**Reported results:**

```text
best_b3_active_inspection_score = 0.985
trace_vs_saliency_rejection = 1.000
delayed_information_gain = 1.000
trace_ablation_inspection_drop = 1.000
delay_ood_inspection_accuracy = 0.975
random_inspection_score = 0.006
```

**Claim:**  
Private delayed trace can guide active inspection in the toy environment.

**Danger:**  
Three models selected the same region on standard episodes.

---

### 3.9 B3.1 Active Inspection Degeneracy Audit

**Question:**  
Is B3 success caused by a shared inspection policy or same-region degeneracy?

**Reported results:**

```text
b31_inspection_audit_score = 0.000
cross_model_inspect_region_match_rate = 1.000
exact_all_model_same_region_rate = 1.000
shared_inspection_policy_usage_rate = 0.000
private_trace_inspection_score_usage_rate = 1.000
inspection_scorer_specificity = 1.000
disagreement_inspection_divergence = 1.000
private_trace_over_non_trace_ratio = 1000000.000
```

**Interpretation:**  
The failure was not shared policy or missing trace usage.  
The standard B3 task had a single dominant inspect target.

---

### 3.10 B3.2 Mechanism-Disambiguating Active Inspection

**Question:**  
Can active inspection be mechanism-disambiguated with family-specific targets?

**Reported results:**

```text
best_b32_mechanism_inspection_score = 0.963
family_specific_inspection_accuracy = 1.000
task_conditioned_switch_accuracy = 1.000
mechanism_disagreement_rate = 1.000
cross_model_same_region_rate = 0.000
family_specific_trace_ablation_drop = 1.000
non_target_family_stability = 1.000
```

**Claim:**  
Private delayed traces support mechanism-disambiguated active inspection under toy diagnostics.

**Boundary:**  
Not real active intelligence or real-world inspection.

---

### 3.11 B4 Delayed Trace-Guided Intervention / Action Selection

**Question:**  
Can private delayed trace guide local intervention?

**Reported results:**

```text
best_b4_intervention_score = 0.985
trace_guided_intervention_accuracy = 1.000
intervention_region_accuracy = 1.000
action_type_accuracy = 1.000
outcome_improvement = 1.000
intervention_vs_inspection_gain = 1.000
wrong_region_penalty_sensitivity = 0.900
trace_ablation_intervention_drop = 1.000
delay_ood_intervention_accuracy = 1.000
```

**Initial claim:**  
Private delayed trace can guide local intervention/action selection.

**Danger:**  
Result needed action-policy degeneracy audit.

---

### 3.12 B4.1 Intervention Degeneracy Audit

**Question:**  
Is B4 action selection real or a fixed action-type shortcut?

**Reported results:**

```text
best_b41_intervention_audit_score = 0.000
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
```

**Interpretation:**  
B4 supported trace-guided intervention-region selection, but not differentiated action-type selection.

**Localized failure:**  

```text
fixed_action_type_rate = 1.000
```

---

### 3.13 B4.2 Action-Type Disambiguation

**Question:**  
Can the model choose different action types when action type matters?

**Added pressure:**

```text
action-type-specific intervention targets
correct-region-wrong-action penalty
family-action mapping stress
action-type counterfactual
fixed-action baseline
action-type ablation
action-type OOD
```

**Reported results:**

```text
best_b42_action_type_score = 0.937
fixed_action_type_rate = 0.500
action_type_accuracy = 1.000
region_accuracy = 1.000
joint_region_action_accuracy = 1.000
correct_region_wrong_action_penalty = 0.900
action_type_counterfactual_sensitivity = 0.900
action_type_ood_accuracy = 1.000
value_leakage_count = 0.000
```

**Updated claim:**  
B4.2 resolves the fixed-action-type shortcut under current toy diagnostics.

**Supported now:**  
Private delayed trace can support both intervention-region selection and differentiated action-type selection in the toy PLOS environment.

**Boundary remains:**  
No real-world control, no robotics capability, no engineering deployment, no natural emergence, no complete mechanism independence.

---

## 4. Current strongest claim after B4.2

The current strongest defensible claim is:

> **In a minimal two-dimensional toy environment, short-horizon checkpoint mechanisms fail under delayed operational dependencies. Trace-bearing substrates with private selectors can support delayed checkpoint, active inspection, mechanism-disambiguated inspection, intervention-region selection, and differentiated action-type selection. B4.2 resolves the fixed-action-type shortcut detected by B4.1 under the current toy diagnostics.**

Short version:

> **B-line has advanced from “can the system see operational traces?” to “can delayed trace guide where to inspect, where to intervene, and what action type to use?”**

Even shorter:

> **The current B-line demonstrates a staged toy diagnostic of W → delayed trace → inspect → intervene → action type.**

---

## 5. What cannot be claimed

Even after B4.2, the project cannot claim:

```text
real-world physical intelligence
robotic control
engineering deployment readiness
human-like intuitive physics
language-free cognition solved
blank-slate emergence
complete mechanism independence
unrestricted causal representation learning
general delayed causality
real slope safety AI
```

The project remains a toy diagnostic ladder.

This is acceptable and should be stated explicitly.

---

## 6. Why the bottleneck shifted

Before B4.2, the bottleneck was:

```text
Can the system select different action types,
or is it using a fixed action-type shortcut?
```

B4.2 resolved this under toy diagnostics.

The new bottleneck is:

```text
Can the operational structure live in a closed loop?
```

More precisely:

```text
Can the system:
1. observe,
2. choose whether to inspect or intervene,
3. inspect to reduce uncertainty,
4. update delayed trace after inspection,
5. intervene based on the updated trace,
6. observe consequences,
7. revise trace after feedback?
```

This shifts the project from static staged decision to closed-loop operation.

---

## 7. New bottleneck: epistemic-pragmatic closed loop

B3 and B4 were separated:

```text
B3: inspect
B4: intervene
```

But a real operational system must decide:

```text
Should I inspect first?
Should I intervene now?
Should I do nothing?
What information is worth paying for?
What action is worth the risk?
How should inspection update the intervention?
How should intervention feedback revise the trace?
```

This is where active inference becomes directly relevant:

```text
inspect = epistemic action
intervene = pragmatic action
```

A closed-loop operational system must balance:

```text
epistemic value + pragmatic value - cost - risk
```

Therefore, the next stage should not be “more intervention.”  
It should be:

```text
B5: Epistemic-Pragmatic Closed-Loop Operation
```

---

## 8. What B5 must test

B5 must test a two-step closed loop:

```text
observe
→ inspect or skip
→ update trace
→ intervene or skip
→ observe consequence
→ revise trace
```

B5 must distinguish:

```text
epistemic value:
  value of information gained by inspection

pragmatic value:
  value of outcome improved by intervention
```

B5 must include:

```text
inspect-vs-intervene timing
trace update after inspection
intervention after update
feedback correction
wrong-inspect penalty
wrong-intervene penalty
planning budget
random / saliency / short-horizon / inspect-always / intervene-immediately / oracle baselines
```

The key claim after B5, if passed, would be:

> **Private delayed operational trace can support a minimal epistemic-pragmatic closed loop in a toy environment.**

This would be stronger than B4.2 because it shows the structure is not only used once, but updated and reused.

---

## 9. External research constraints still to absorb

B4.2 does not remove the external blind spots. The next stages still need:

### 9.1 Slot Attention / object-centric learning

Borrowed constraint:

```text
binding competition and slot exchangeability
```

Missing gate:

```text
Trace Slot Competition Test
```

Why it matters:

```text
current trace families are still partly structured by us.
future systems should allocate trace slots among competing candidates.
```

---

### 9.2 Active inference

Borrowed constraint:

```text
epistemic value vs pragmatic value
```

Missing gate:

```text
Epistemic-Pragmatic Split Test
```

B5 should directly absorb this.

---

### 9.3 CausalTriplet / actionable counterfactuals

Borrowed constraint:

```text
not all variables are actionable
not all counterfactuals are controllable
```

Missing gate:

```text
Actionability Mask
```

Need fields:

```text
observable
inspectable
intervenable
directly controllable
indirectly controllable
unsafe
irreversible
costly
```

---

### 9.4 IntPhys / intuitive physics benchmark

Borrowed constraint:

```text
minimal matched sets
surface-statistic control
object permanence
continuity
```

Missing gate:

```text
Matched Minimal Set Audit
```

---

### 9.5 PHYRE / physical reasoning benchmark

Borrowed constraint:

```text
external 2D physical reasoning benchmark translation
sample-efficient generalization
```

Missing later:

```text
External Benchmark Translation Layer
```

---

### 9.6 Identifiability research

Borrowed constraint:

```text
same behavior may have multiple equivalent internal explanations
```

Missing gate:

```text
Identifiability Audit
```

Needed questions:

```text
Is delayed trace uniquely identifiable?
Is intervention target unique or one of many equivalent targets?
Can recurrent / field / schema mechanisms be distinguished by available interventions?
```

---

## 10. Updated route split

### 10.1 Discriminator route

Goal:

```text
Make PLOS-Test a strong diagnostic benchmark.
```

Next discriminator modules:

```text
B5 Epistemic-Pragmatic Closed-Loop Operation
Identifiability Audit
Actionability Mask
Planning Budget Gate
Uncertainty Calibration Gate
Trace Slot Competition Test
External Benchmark Translation Layer
```

This route strengthens scientific credibility.

---

### 10.2 Generator route

Goal:

```text
Make a minimal substrate grow operational structure,
instead of hand-assembling trace families and selectors.
```

Candidate pressures:

```text
compression pressure
prediction pressure
delayed trace pressure
inspect cost
intervention cost
wrong-region penalty
wrong-action penalty
feedback correction
memory bottleneck
slot competition
local edit loss
```

The generator route should begin only after B5 or after a separate design freeze.

---

## 11. Updated theory statement

The theory now stands as:

> **Operational structure is an internal, reusable, locally intervenable intermediary formed from continuous world dynamics. It becomes meaningful only when it supports multiple operations—prediction, inspection, intervention-region selection, action-type selection, and eventually closed-loop update—while surviving false-positive audits and budget constraints.**

Current achieved layers:

```text
W → O₁ delayed trace
O₁ → checkpoint
O₁/O₂ → active inspection
O₂ → intervention region
O₂ → action type
```

Current missing layer:

```text
O₂ → closed-loop update
```

---

## 12. Updated failure conditions

The W → O₁ → O₂ → L framework would be weakened if:

1. A first-order predictor without separable O passes all checkpoint, inspection, intervention, action-type, OOD, and closed-loop gates.

2. Local ablation/edit of O does not produce predicted local behavioral change.

3. Trace-guided policies fail to outperform saliency, short-horizon, random, fixed-action, or inspect-only baselines under matched conditions.

4. Mechanism-disambiguating episodes still collapse to identical decisions.

5. B5 closed-loop update fails: inspection does not improve trace, updated trace does not improve intervention, and feedback does not revise future decisions.

6. Identifiability audit shows the supposed O is indistinguishable from simpler non-structural heuristics.

7. A language-only system passes B-line gates without access to W-level continuous facts or operational state.

---

## 13. Recommended immediate next step

The immediate next stage should be:

```text
B5: Epistemic-Pragmatic Closed-Loop Operation
```

Not:

```text
more B4 hardening
larger action space
more complex video world
LLM integration
real robotics
```

B5 should be small and surgical:

```text
two-step closed loop
one inspect opportunity
one intervention opportunity
trace update after inspection
feedback update after intervention
explicit epistemic/pragmatic value split
strict baselines
planning budget
```

---

## 14. Summary after B4.2

Before B4.2:

```text
current bottleneck = action-type disambiguation
```

After B4.2:

```text
current bottleneck = closed-loop trace update and epistemic-pragmatic action selection
```

Final V2 summary:

> **The project has reached the point where one-shot operational use is largely demonstrated in the toy setting: delayed trace can guide prediction, inspection, intervention region, and action type. The next scientific question is whether the trace can be updated and reused in a closed loop where the system must decide when to gather information and when to act.**
