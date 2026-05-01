# PROJECT_THEORY_FREEZE_V1.md

**Project:** Relation Internalization Program / Pre-Linguistic Operational Structure  
**Freeze date:** 2026-05-01  
**Status:** Historical V1 theory freeze before B4.2 was incorporated. Superseded by `PROJECT_THEORY_FREEZE_V2.md`.  
**Purpose:** Freeze the conceptual core, evidence chain, current strongest claim, claim boundary, blind spots, and next-stage split into discriminator route and generator route.

---

## 0. One-sentence freeze

This project has moved from asking whether a model “understands relations” to asking whether a system can form an internal operational structure from continuous world dynamics, use that structure as a causal intermediary for prediction, inspection, and intervention, and survive false-positive audits such as shared selectors, fixed action shortcuts, leakage, weak baselines, and nonspecific ablations.

The current strongest framing is:

> **Operational structure internalization is not language-level relation description. It is the formation of a non-linguistic internal intermediary O from continuous world W, where O supports prediction, delayed trace, active inspection, local intervention, and eventually action-type selection under budget, cost, and feedback constraints.**

---

## 1. Original starting point: relation internalization

The original project asked:

> Does a system merely predict outputs, or does it internalize relations?

Early A-line relation internalization treated relation structure as something like an internal relation table:

```text
support examples / observations
→ internal relation representation R_hat
→ transfer
→ counterfactual
→ local edit
→ audit
→ budgeted inspect
```

The original gates were:

1. **Symbol transfer**: Can the system transfer a relation under randomized symbols?
2. **Support-conditioned binding**: Does the same query change with different support examples?
3. **Local edit**: Can one relation be edited without global behavior collapse?
4. **Exact audit**: Can the system state which relation chain is uncertain or active?
5. **Budgeted inspect**: Can it choose what to inspect under limited observation budget?

This was a strong improvement over plain behavior accuracy, because it separated:

```text
answering correctly
from
holding an editable, inspectable, actionable relation structure
```

However, it remained too language-adjacent and table-adjacent. It risked treating relation structure as a symbolic object first, then asking whether the model possessed it.

---

## 2. Key correction: facts happen before language covers them

The project changed direction after the correction:

> Facts happen first; language symbols come later and cover them.

In physical reality, balls move, collide, occlude, and reappear before anyone labels them as “objects,” “events,” or “causes.” A slope moves, cracks open, pore pressure rises, drainage changes, and displacement accumulates before engineers describe these processes in language.

This led to the W → O₁ → O₂ → L revision.

---

## 3. Revised framework: W → O₁ → O₂ → L

### 3.1 W: continuous world / fact occurrence layer

W is the continuous world process itself.

Examples:

```text
ball rolls
object occludes
collision redirects motion
local force field changes trajectory
water infiltrates
pore pressure rises
slope displacement accumulates
crack expands
```

W does not require symbols, language, or model interpretation.

W is not intelligence.  
W is the source process that may or may not be internalized by a system.

---

### 3.2 O₁: trajectory schema / delayed trace / pre-linguistic operational trace

O₁ is the first internal operational compression of W.

It includes:

```text
object persistence
motion continuity
occlusion continuation
crossing identity preservation
collision event boundary
delayed trace
checkpoint
short-term and delayed influence
```

O₁ is not yet a language relation table.  
It is closer to a trajectory schema or operational trace.

O₁ answers:

```text
what continues?
what changes?
what was delayed?
where is the hidden influence?
what must be remembered across time?
```

---

### 3.3 O₂: actionable operational structure

O₂ is O₁ made usable for action.

It includes:

```text
inspect target
intervention target
action region
action type
cost-aware action value
wrong-region penalty
wrong-action penalty
budgeted selection
```

O₂ answers:

```text
where should I look?
where should I intervene?
what kind of action should I apply?
what happens if I choose the wrong region?
what happens if I choose the right region but wrong action type?
```

---

### 3.4 L: language cover / audit layer

L is the language layer that describes, reports, audits, or explains W/O.

Examples:

```text
“Rainfall caused pore pressure increase.”
“Object A remained behind the occluder.”
“The uncertain link is displacement → crack.”
“The model should inspect the delayed trace region.”
```

Language is useful for audit and human communication, but it is not the internalization itself.

A system may produce fluent L without robust O.

Therefore:

```text
L → L is not enough.
W → O₁ → O₂ → L is the target.
```

---

## 4. Revised hierarchy of systems

### 4.1 Zero-order: fact layer

The world evolves. No internal model is implied.

### 4.2 First-order: reaction / prediction

A system maps input history to output:

```text
input stream → prediction / reaction
```

It may be strong at prediction but lacks a separable operational intermediary.

Examples:

```text
pixel predictor
trajectory memory
ordinary RNN / Transformer world model
LLM prompt-only response
```

First-order systems may predict, but they often fail under:

```text
occlusion
crossing
delayed influence
local intervention
budgeted inspection
action-type disambiguation
```

### 4.3 Second-order: operational structure internalization

A system forms O as a separable, causally used intermediary:

```text
W → O → prediction / inspect / intervention / action
```

Second-order systems must satisfy:

```text
persistence
selectivity
locality
actionability
budgeted use
ablation specificity
OOD robustness
```

### 4.4 Third-order: meta-operational correction

A system not only updates O, but revises how it forms O.

It asks:

```text
Am I using the wrong cue to form object identity?
Do I over-rely on saliency?
Do I confuse short-horizon checkpoints with delayed trace?
Do I repeatedly choose fixed action types?
Should I change my own structure-forming rule?
```

Current project status: B-line has advanced second-order diagnostics, but not yet true third-order meta-correction.

---

## 5. A-line vs B-line

### 5.1 A-line: symbolic relation internalization

A-line is still valuable. It is best for:

```text
relation table
support-conditioned binding
counterfactual relation edit
exact audit
budgeted variable inspection
engineering audit and reporting
```

A-line asks:

> Does the system hold a relation structure that can be transferred, edited, audited, and used to inspect missing variables?

It is appropriate for AI agent auditing, symbolic relation systems, and engineered decision support.

### 5.2 B-line: pre-linguistic operational structure internalization

B-line is deeper and more physical.

It asks:

> Can a system form operational structure before language, directly from continuous world dynamics?

B-line is appropriate for:

```text
object persistence
event boundary
collision relation
delayed trace
active inspection
local intervention
action-type selection
feedback update
```

A-line and B-line are complementary:

```text
A-line: relation table / symbolic audit
B-line: operational trace / action structure
```

The long-term target is not A or B alone, but:

```text
W → O₁ → O₂ → L
```

---

## 6. Evidence ladder: PLOS / B1 / B2 / B3 / B4

This section freezes the evidence chain as currently understood from the project runs reported so far.

---

### 6.1 PLOS v1: Pre-Linguistic Operational Structure Test

**Question:**  
Can any candidate model pass a minimal test for pre-linguistic operational checkpoint structure in a 2D toy world?

**Result:**  
`flow_checkpoint_model` emerged as the initial PLOS candidate.

**Supported claim:**  
A short-horizon flow/checkpoint model can capture some local operational checkpoint signals.

**Unsupported claim:**  
It does not prove delayed causality, trace memory, active inspection, or intervention ability.

---

### 6.2 B1.1: Flow-Checkpoint Reviewer Hardening

**Question:**  
Is `flow_checkpoint_model` merely exploiting saliency or easy local shortcuts?

**Attack set included:**

```text
dynamic decoy checkpoint
delayed checkpoint
competing checkpoints
checkpoint relocation OOD
causal deletion vs visual deletion
anti-prior world
```

**Result:**  
The model survived some local attacks but failed delayed checkpoint.

**Supported claim:**  
Short-horizon checkpoint is not completely trivial.

**Key failure:**  
Delayed checkpoint accuracy collapsed.

**Interpretation:**  
Short-horizon checkpoint is insufficient for delayed operational structure.

---

### 6.3 B2: Delayed Operational Checkpoint Substrate

**Question:**  
What substrate can handle delayed checkpoint?

**Added models:**

```text
recurrent_flow_checkpoint_model
field_memory_model
schema_memory_model
```

**Result:**  
Trace-bearing substrates solved delayed checkpoint.

**Supported claim:**  
Delayed checkpoint requires a trace-bearing path.

**Unsupported claim:**  
At this point, it did not prove independent mechanisms.

---

### 6.4 B2.1: Trace-Bearing Substrate Hardening

**Question:**  
Can trace-bearing models survive stronger trace attacks?

**Attacks included:**

```text
false delayed trace
trace swap
trace deletion specificity
multi-source trace conflict
noisy delayed trace
trace length extrapolation
trace compression pressure
```

**Result:**  
The trace-bearing models passed strongly.

**Initial danger:**  
All three models had identical or near-identical scores.

**Supported claim:**  
Trace-bearing path is useful under current gates.

**Unsupported claim:**  
The three trace-bearing models were not yet proven independent.

---

### 6.5 B2.1a: Trace Hardening Score Degeneracy Audit

**Question:**  
Why did three different trace-bearing models receive identical scores?

**Audit included:**

```text
per-attack breakdown
per-seed breakdown
predicted region distribution
ground-truth leakage check
intervention applicability
random-trace baseline
oracle baseline
trace-family ablation
no-trace ablation
```

**Result:**  
No ground-truth key leakage, no random passability, no inactive intervention.  
But all models made identical per-episode predictions.

**Interpretation:**  
B2/B2.1 supported trace-bearing path usefulness, not independent recurrent/field/schema mechanisms.

---

### 6.6 B2.2: Trace Selector Disentanglement

**Question:**  
Do the three trace-bearing models use independent selectors, or a shared trace selector?

**Result:**  
Conservative failure.

**Interpretation:**  
B2/B2.1 success was better interpreted as shared trace-selector success.

**Supported claim:**  
Trace path useful.

**Unsupported claim:**  
Independent trace mechanisms.

---

### 6.7 B2.3: Private Trace Selector Construction

**Question:**  
Can recurrent / field / schema models be reconstructed with private selectors?

**Key reported results:**

```text
shared_selector_usage_rate = 0.000
model_private_score_usage_rate = 1.000
cross_model_exact_prediction_match_rate = 0.000
disagreement_episode_divergence = 1.000
b2_delayed_score = 1.000
b21_trace_hardening_score = 0.960
b23_private_selector_score = 0.972
```

**Supported claim:**  
Shared selector explanation was reduced.  
Recurrent / field / schema trace-bearing paths showed partial mechanism separation under toy diagnostics.

**Unsupported claim:**  
Complete independence, natural emergence, or general delayed causality.

---

### 6.8 B3: Delayed Trace-Guided Active Inspection

**Question:**  
Can private delayed trace guide active inspection under budget?

**Key reported results:**

```text
best_b3_active_inspection_score = 0.985
trace_vs_saliency_rejection = 1.000
delayed_information_gain = 1.000
trace_ablation_inspection_drop = 1.000
delay_ood_inspection_accuracy = 0.975
random_inspection_score = 0.006
```

**Supported claim:**  
Private delayed trace can guide active inspection under current toy diagnostics.

**Danger:**  
Three models again showed identical or near-identical B3 performance.

---

### 6.9 B3.1: Active Inspection Degeneracy Audit

**Question:**  
Is B3 success caused by shared inspection policy or same-region degeneracy?

**Key reported results:**

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
B3 remained valid as trace-guided inspection path evidence, but standard B3 episodes had a single dominant inspect target.

**Failure localized to:**  
Task not mechanism-disambiguating enough.

---

### 6.10 B3.2: Mechanism-Disambiguating Active Inspection

**Question:**  
Can recurrent / field / schema traces guide different active inspection strategies when family-specific targets differ?

**Key reported results:**

```text
best_b32_mechanism_inspection_score = 0.963
family_specific_inspection_accuracy = 1.000
task_conditioned_switch_accuracy = 1.000
mechanism_disagreement_rate = 1.000
cross_model_same_region_rate = 0.000
family_specific_trace_ablation_drop = 1.000
non_target_family_stability = 1.000
```

**Supported claim:**  
B3.2 reduced B3.1 same-region degeneracy and supported partial mechanism-disambiguated active inspection under current toy diagnostics.

**Unsupported claim:**  
Real active intelligence, real-world inspection, complete mechanism independence.

---

### 6.11 B4: Delayed Trace-Guided Intervention / Action Selection

**Question:**  
Can delayed trace guide local intervention region and action selection?

**Key reported results:**

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

**Supported claim before audit:**  
Private delayed trace can guide minimal local intervention in the toy world.

**Danger:**  
The result was too clean; action-policy shortcuts needed auditing.

---

### 6.12 B4.1: Intervention Degeneracy Audit

**Question:**  
Does B4 really support differentiated action mechanisms?

**Key reported results:**

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
B4 supports trace-guided intervention-region selection, but B4.1 found a fixed action-type shortcut.

**Current B4-level claim must be narrowed:**

```text
Supported:
  private delayed trace can guide intervention-region selection.

Not yet supported:
  differentiated action-type selection.
```

---

### 6.13 B4.2: Action-Type Disambiguation

**Status:**  
Designed as the next repair layer. If not yet run, this remains pending.

**Purpose:**  
Directly attack B4.1’s fixed action-type failure.

**Required additions:**

```text
action-type-specific intervention targets
correct-region-wrong-action penalty
family-action mapping stress
action-type counterfactual
fixed-action baseline
action-type ablation
action-type OOD
```

**Core question:**  
Can private delayed trace guide not only where to intervene, but which action type to use?

**Potential supported claim if passed:**  
Private delayed trace supports differentiated action-type selection in the toy PLOS environment.

**Failure meaning:**  
Current system can find where to intervene, but cannot decide how to intervene.

---

## 7. Current strongest claim

The current strongest defensible claim is:

> **In a minimal two-dimensional toy environment, short-horizon checkpoint mechanisms are insufficient for delayed operational dependencies. Trace-bearing substrates can solve delayed checkpoint. After removing shared selector degeneracy, private recurrent / field / schema trace paths can support delayed trace, active inspection, mechanism-disambiguated inspection, and intervention-region selection. However, differentiated action-type selection remains unresolved unless B4.2 passes.**

Shorter version:

> **B-line has moved from prediction to delayed trace, from trace to active inspection, and from inspection to intervention-region selection. The current unresolved bottleneck is action-type disambiguation and, more broadly, closed-loop update.**

---

## 8. Current claim boundaries

The project does **not** currently prove:

```text
real-world physical intelligence
robotic control
engineering deployment readiness
human-like intuitive physics
language-free cognition solved
blank-slate emergence
complete mechanism independence
general causal representation learning
unrestricted delayed causality
```

The project currently supports only toy-diagnostic claims.

This is not a weakness if written honestly. It is the correct scientific boundary.

---

## 9. What this project is really becoming

It is not merely a model-building project.

It is becoming:

```text
a staged diagnostic ladder for pre-linguistic operational structure
```

Its real contribution is the repeated conversion of vague claims into falsifiable gates:

```text
prediction is not trace
trace is not private mechanism
private trace is not active inspection
active inspection is not intervention
intervention region is not action-type selection
high score is not mechanism unless audited
```

This “false-positive ladder” is the project’s strongest methodological contribution.

---

## 10. External research absorbed into the freeze

This project should not pretend that object-centric learning, physical reasoning, causal representation, or active perception are new. They are established neighboring fields. The project’s contribution is to organize their insights into an operational-structure diagnostic sequence.

---

### 10.1 Slot Attention / object-centric learning

Slot Attention extracts object-centric representations from perceptual input through exchangeable slots that specialize via competitive attention, enabling generalization to unseen compositions.

**Borrowed lesson:**

```text
binding matters
slot competition matters
exchangeability matters
multi-object / multi-trace assignment matters
```

**Our missing gate:**

```text
Trace Slot Competition Test
```

Needed tests:

```text
multiple trace candidates
trace candidate permutation
trace merge/split
variable number of trace slots
slot identity persistence under occlusion
```

**Why it matters:**  
Current B-line often gives recurrent / field / schema traces as defined families. A more natural system must bind competing trace candidates without us pre-assigning them.

---

### 10.2 Active inference / epistemic value

Active inference decomposes policy value into epistemic value and extrinsic/pragmatic value: actions may be chosen either to gain information or to achieve preferred outcomes.

**Borrowed lesson:**

```text
inspect is epistemic action
intervene is pragmatic action
a mature agent must trade off both
```

**Our missing gate:**

```text
Epistemic-Pragmatic Split Test
```

Needed decomposition:

```text
action_value = epistemic_gain + pragmatic_gain - cost
```

**Why it matters:**  
B3 and B4 should not be blurred. B3 asks “what should I check to reduce uncertainty?” B4 asks “what should I do to change future outcome?” B5 must explicitly combine both.

---

### 10.3 CausalTriplet / intervention-centric causal representation

CausalTriplet emphasizes actionable counterfactuals, object-level variables, OOD robustness, and independent causal mechanisms.

**Borrowed lesson:**

```text
not every variable is intervenable
not every counterfactual is actionable
intervention must respect actionability
OOD robustness matters
```

**Our missing gate:**

```text
Actionability Mask
```

Needed fields:

```text
observable?
inspectable?
directly intervenable?
indirectly intervenable?
controllable?
cost?
risk?
reversibility?
```

**Why it matters:**  
Toy actions currently risk making every region/action available. Real operational structure must know which parts of the world can be inspected, changed, or only inferred.

---

### 10.4 IntPhys / intuitive physics benchmarks

IntPhys emphasizes object permanence, continuity, minimal matched sets, and controlled difficulty through occlusion duration, object number, and movement.

**Borrowed lesson:**

```text
minimal matched sets reduce bias
one principle should be tested at a time
occlusion and object permanence are foundational
```

**Our missing gate:**

```text
Matched Minimal Set Audit
```

Needed checks:

```text
possible/impossible paired episodes
same surface statistics
only operational principle differs
difficulty controlled parametrically
```

**Why it matters:**  
Our toy generator must avoid accidental cues. IntPhys reminds us to construct controlled matched sets.

---

### 10.5 PHYRE / physical reasoning benchmarks

PHYRE is a 2D physical reasoning benchmark designed to test sample-efficient generalization across classical mechanics puzzles.

**Borrowed lesson:**

```text
external physical benchmark translation matters
sample efficiency matters
generalization across puzzle families matters
```

**Our missing gate:**

```text
External Benchmark Translation Layer
```

Possible later mapping:

```text
PLOS object persistence → IntPhys-style permanence
PLOS action/intervention → PHYRE-style physical puzzle action
PLOS trace-guided action → physical intervention planning
```

**Why it matters:**  
Eventually PLOS-Test should not live only inside its own generator.

---

### 10.6 CausalVerse / high-fidelity causal simulation

CausalVerse-style work emphasizes configurable simulations with ground-truth causal mechanisms, variables, interventions, and temporal dependencies.

**Borrowed lesson:**

```text
evaluation needs ground-truth mechanisms
but also visual/dynamic complexity
configurability matters
```

**Our missing gate:**

```text
Mechanism Ground-Truth Audit
```

Needed checks:

```text
what is simulator-defined?
what is learned?
what is evaluator-imposed?
what is unique vs equivalent?
```

---

## 11. Core blind spots after this freeze

The main blind spots are:

1. **Identifiability**  
   We have many gates, but not yet a formal audit of whether trace / checkpoint / action mechanisms are uniquely identifiable or only equivalent explanations.

2. **Actionability mask**  
   We have inspect/intervene actions, but not a principled distinction between observable, inspectable, intervenable, controllable, costly, risky, and irreversible variables.

3. **Epistemic-pragmatic decomposition**  
   B3 and B4 are separated experimentally, but not yet unified under a value decomposition.

4. **Coordinate frame transfer**  
   Current region_id grid is useful but image-centric. We need allocentric / object-relative / action-relative coordinate tests.

5. **Trace slot competition**  
   Current trace families are partly hand-structured. A stronger system should allocate trace slots among multiple candidates.

6. **Planning budget**  
   We often allow enumeration over candidates. A real operational structure should work under compute/search limits.

7. **Uncertainty calibration**  
   The system should know where it is uncertain, not just choose a high-value region.

8. **Closed-loop update**  
   Current stages are mostly single-pass. Operational structure should update after inspect/intervene feedback.

9. **Natural emergence**  
   The current project is mainly a discriminator/evaluator route. It has not yet shown a minimal system self-generating O under pressure.

10. **Action-type disambiguation**  
    B4.1 exposed a fixed action-type shortcut. B4.2 must resolve this before B5.

---

## 12. Current theory: operational structure as a reusable causal intermediary

The project should now define operational structure as:

> **An internal intermediary O formed from continuous world W, which is reused across prediction, inspection, intervention, and action selection; whose local deletion, edit, or ablation causes predictable local behavioral changes; and whose use remains valid under budget, OOD, and false-positive audits.**

This definition implies that a behavior is not enough. The system must show:

```text
cross-task reuse
locality
ablation specificity
counterfactual sensitivity
budgeted use
OOD transfer
auditability or inspectability
```

---

## 13. Discriminator route vs generator route

The next phase must split into two routes.

---

### 13.1 Route A: discriminator / benchmark route

Goal:

```text
Define what counts as operational structure.
Detect false positives.
Give any model a staged diagnostic report.
```

Next additions should be:

```text
B4.2 Action-Type Disambiguation
Identifiability Audit
Actionability Mask
Epistemic-Pragmatic Split Test
Planning Budget Gate
Uncertainty Calibration Gate
Trace Slot Competition Test
External Benchmark Translation Layer
```

This route is lower risk and directly supports a paper/benchmark.

---

### 13.2 Route B: generator / emergence route

Goal:

```text
Make a minimal half-egg substrate grow O by pressure,
instead of hand-designing O.
```

Candidate pressures:

```text
compression pressure
prediction pressure
delayed trace pressure
inspection cost
intervention cost
wrong-region penalty
wrong-action penalty
feedback update
memory bottleneck
slot competition
local edit loss
```

Minimal target:

```text
continuous toy world
small trainable substrate
no hand-coded relation table
no hand-coded trace selector
learned trace memory
learned inspect target
learned intervention target
learned action type
closed-loop feedback update
```

This route is higher risk but closer to the original question:

> How does operational structure emerge?

---

## 14. Recommended immediate next steps

### Step 1: Finish or run B4.2 if not already done

B4.2 is necessary because B4.1 found:

```text
fixed_action_type_rate = 1.000
action_type_shift_after_trace_ablation = 0.000
```

B4.2 should not be skipped.

---

### Step 2: Add Identifiability Audit

A theory without identifiability analysis is vulnerable.

Ask:

```text
Which structures are uniquely identified?
Which are equivalent explanations?
What interventions are required to distinguish them?
```

---

### Step 3: Add Epistemic-Pragmatic Split

Before B5, define:

```text
inspect = epistemic action
intervention = pragmatic action
```

and test:

```text
when to inspect
when to intervene
when to do both
when not to act
```

---

### Step 4: Define Actionability Mask

Every region or latent variable should be annotated as:

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

This prepares the project for engineering relevance.

---

### Step 5: Start generator-route planning only after theory freeze

Do not jump immediately to a large learned model.

First define the smallest system that could grow:

```text
trace
checkpoint
inspect target
intervention target
action type
update rule
```

under pressure.

---

## 15. Theoretical failure conditions

The W → O₁ → O₂ → L framework would be weakened or falsified if:

1. A first-order predictor without separable O consistently passes prediction, inspection, intervention, action-type, OOD, and ablation-specificity gates.

2. Local ablation/edit of O fails to produce local behavioral change, while behavior remains explainable by direct input-output heuristics.

3. Trace-guided inspection/intervention does not outperform saliency, short-horizon, random, or fixed-action baselines under properly matched tasks.

4. Multiple supposedly distinct mechanisms remain indistinguishable under mechanism-disambiguating episodes.

5. Action outcomes are explained better by fixed policies than by trace-conditioned structures.

6. Identifiability audit shows that the supposed O is not distinguishable from simpler equivalent representations under the available observations and interventions.

7. A language-only model passes all B-line operational gates without access to W-level continuous facts or non-linguistic operational state.

These are useful failure conditions. They keep the framework scientific rather than purely philosophical.

---

## 16. Paper-level positioning

A possible paper framing:

> Existing work has studied object-centric representation, intuitive physics, causal representation learning, active perception, and physical reasoning benchmarks. This project does not claim to invent these components. Its contribution is to organize them into a staged diagnostic ladder for pre-linguistic operational structure: from continuous world dynamics to delayed trace, private selector, active inspection, local intervention, and action-type disambiguation, with explicit audits for shared selectors, fixed action shortcuts, leakage, weak baselines, nonspecific ablations, and mechanism indistinguishability.

---

## 17. Current final summary

The project has advanced from:

```text
Does the model understand relations?
```

to:

```text
Can a system form a non-linguistic operational intermediary from continuous world dynamics,
and use it under delay, budget, inspection, intervention, and action constraints?
```

The current best answer is:

```text
In a controlled toy world, yes for delayed trace, private selector, active inspection,
mechanism-disambiguated inspection, and intervention-region selection.

Differentiated action-type selection remains the current unresolved bottleneck
unless B4.2 passes.
```

The final long-term question is:

```text
What substrate and pressure conditions cause such an operational structure to emerge,
rather than be hand-assembled?
```

---

## 18. Source notes

This theory freeze borrows constraints and language from several neighboring research streams:

1. Slot Attention / object-centric learning: object-centric slots, exchangeability, competitive binding, and composition generalization.
2. Active inference: decomposition of policy value into epistemic and extrinsic/pragmatic value.
3. CausalTriplet: actionable counterfactuals, object-level causal variables, OOD robustness, and independent causal mechanisms.
4. IntPhys: object permanence, minimal matched sets, and controlled intuitive physics tests.
5. PHYRE: 2D physical reasoning, sample efficiency, and generalization across physical puzzles.
6. CausalVerse-style causal simulation: configurable ground-truth causal mechanisms, variables, interventions, and temporal dependencies.

Reference links are intentionally placed here rather than in the main claim body to keep the project claim self-contained:

- Slot Attention, NeurIPS 2020: https://papers.nips.cc/paper_files/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html
- Active inference and epistemic value: https://research-portal.uea.ac.uk/en/publications/active-inference-and-epistemic-value
- CausalTriplet, PMLR 2023: https://proceedings.mlr.press/v213/liu23a.html
- IntPhys benchmark description: https://intphys.cognitive-ml.fr/benchmark/description.html
- PHYRE benchmark, NeurIPS 2019: https://papers.nips.cc/paper/8752-phyre-a-new-benchmark-for-physical-reasoning
- CausalVerse benchmark repository: https://github.com/CausalVerse/CausalVerseBenchmark
