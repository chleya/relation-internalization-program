# Relation Internalization Project Summary

Date: 2026-04-28

## 0. One-Sentence Summary

This project builds a runnable diagnostic chain for testing whether a system has internalized relations, rather than merely fitting input-output mappings, memorizing contexts, following surface cues, or producing plausible review text.

The current evidence chain is:

```text
food-world explicit relation table
-> neural hidden-state relation probe
-> neural-to-table extraction
-> slope-engineering relation-chain review toy
```

The key claim is deliberately narrow:

```text
A relation-internalized system should expose usable internal relations that are transferable, counterfactual, editable, auditable, and action-guiding.
```

The project does not claim real-world engineering safety capability.

---

## 1. Project Map

### 1.1 `F:\relation-internalization-test`

Purpose:

```text
Define and test the minimal relation-internalization concept in a food/poison toy world.
```

Core question:

```text
Can an agent use the true texture/wet -> resource relation under OOD, reversal, counterfactual, and internal edit tests?
```

Main outputs:

```text
results/summary.csv
figures/*.png
reports/V1_RESEARCH_REPORT.md
reports/auto_report.md
```

Test status:

```text
17 passed
```

### 1.2 `F:\neural-relation-probe`

Purpose:

```text
Test whether a neural hidden state encodes and causally uses the relation.
```

Core question:

```text
Is relation information merely probe-readable, or behaviorally necessary?
```

Main outputs:

```text
results/summary_modes.csv
results/extraction_summary.csv
figures/neural_probe_modes.png
figures/extraction_modes.png
reports/AUTO_REPORT.md
```

Test status:

```text
7 passed
```

### 1.3 `F:\slope-relation-toy`

Purpose:

```text
Move the relation-internalization diagnostic into a slope-engineering review toy world.
```

Core question:

```text
Can an AI review system bind actions to engineering relation chains, rather than surface labels or generic safety language?
```

Main outputs:

```text
results/summary.csv
figures/slope_scores.png
reports/auto_report.md
reports/review_example.json
reports/self_audit.md
```

Test status:

```text
12 passed
```

---

## 2. Core Concept

The working definition of relation internalization is:

```text
External relations become internal usable structures.
```

In this project, "usable" means the relation structure supports:

- OOD transfer
- spurious-cue resistance
- counterfactual intervention
- internal editing
- relation audit
- action-point consistency
- engineering review explanation

This definition intentionally rejects weaker substitutes:

```text
high prediction accuracy != relation internalization
probe readability != relation internalization
generic review language != relation internalization
context memory != relation internalization
```

---

## 3. Project 1: `relation-internalization-test`

## 3.1 Environment

The first project uses a food/poison world.

Context:

```text
texture: A / B / C
wet: dry / wet
color: red / blue
odor: strong / weak
```

True relation:

```text
texture=A AND wet=dry -> food
texture=A AND wet=wet -> poison
texture=B             -> poison
texture=C             -> neutral
```

Surface features:

```text
color, odor
```

These are deliberately spurious.

## 3.2 Models

Implemented agents include:

```text
majority
memory
fitting
predictive
decision_tree
relation
wide_relation
robust_wide_relation
```

The main relation model maintains an internal editable table:

```text
condition -> resource distribution
```

Key interface:

```python
def edit_rule(condition: dict, new_outcome: str) -> bool:
    ...

def describe_relations() -> list[dict]:
    ...
```

## 3.3 Key Code: Relation Inference

The intended scoring logic is:

```python
score(resource) = sum over matching rules:
    confidence * specificity(condition) * P(resource | rule)
```

Representative logic:

```python
def infer_resource(self, context: dict) -> str:
    scores = {resource: 0.0 for resource in RESOURCES}
    for rule in self.matching_rules(context):
        if rule.support < self.min_support:
            continue
        specificity = len(rule.condition)
        weight = (rule.confidence ** self.confidence_weight) * (
            specificity ** self.specificity_weight
        )
        total = sum(rule.counts.values())
        for resource in RESOURCES:
            scores[resource] += weight * (rule.counts.get(resource, 0) / total)
    return max(scores, key=scores.get)
```

Key action policy:

```python
def act(self, context: dict) -> str:
    return "eat" if self.infer_resource(context) == "food" else "avoid"
```

Key edit behavior:

```python
edit_rule({"texture": "A", "wet": "dry"}, "poison")
```

This should change behavior from:

```text
eat -> avoid
```

on the target context.

## 3.4 Gated Score

A major lesson from `svt_agents` was adopted:

```text
high average score is not enough;
critical gates must all pass.
```

The gated internalization score is zero unless all core gates pass:

```text
ood_success >= 0.9
spurious_resource_accuracy >= 0.8
counterfactual_accuracy >= 0.9
edit_resource_success >= 0.9
edit_reversal_success >= 0.9
relation_table_alignment >= 0.9
relation_shuffle_drop >= 0.25
```

## 3.5 Main Finding

The restricted `relation` model and `robust_wide_relation` pass the gates.

Important negative result:

```text
wide_relation stores correct relations but can still choose spurious rules.
```

This means:

```text
having a correct relation somewhere inside the system is not enough;
the system must select and use the right relation under pressure.
```

---

## 4. Project 2: `neural-relation-probe`

## 4.1 Purpose

The second project asks:

```text
Can a neural model encode relation information, and does behavior causally depend on it?
```

This project was motivated by the key warning from `probe_to_boundary_gnn`:

```text
probe readability != causal use
```

## 4.2 Minimal Neural Setup

The model is a lightweight sklearn MLP:

```python
def train_mlp(x: np.ndarray, y: np.ndarray, seed: int = 0) -> MLPClassifier:
    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.01,
        max_iter=600,
        random_state=seed,
    )
    clf.fit(x, y)
    return clf
```

Hidden state extraction:

```python
def hidden_states(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    hidden = x @ model.coefs_[0] + model.intercepts_[0]
    return np.maximum(hidden, 0.0)
```

## 4.3 Probe and Control

Linear probe:

```python
def train_probe(hidden: np.ndarray, labels: np.ndarray, seed: int = 0) -> LogisticRegression:
    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe.fit(hidden, labels)
    return probe
```

Random-label control:

```python
def random_label_control(hidden: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    probe = train_probe(hidden, shuffled, seed=seed)
    return probe_accuracy(probe, hidden, shuffled)
```

This checks whether the probe is extracting real structure or just overfitting.

## 4.4 Causal Intervention

The first intervention, dimension zeroing, was too weak:

```text
relation_dimension_zero_drop ~= 0.013
```

The stronger intervention removes the linear relation subspace:

```python
def remove_probe_subspace(hidden: np.ndarray, probe: LogisticRegression) -> np.ndarray:
    weights = probe.coef_
    _, singular_values, vh = np.linalg.svd(weights, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-8))
    if rank == 0:
        return hidden.copy()
    basis = vh[:rank].T
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    projected = centered @ basis @ basis.T
    return hidden - projected
```

Then task drop is measured:

```python
def subspace_intervention_drop(model, x, y, probe) -> float:
    hidden = hidden_states(model, x)
    base_pred = predict_from_hidden(model, hidden)
    edited = remove_probe_subspace(hidden, probe)
    edited_pred = predict_from_hidden(model, edited)
    return accuracy_score(y, base_pred) - accuracy_score(y, edited_pred)
```

## 4.5 Base vs Shortcut

Two training modes were compared:

```text
base:
model must use texture/wet relation.

shortcut:
color/odor surface cues are strongly tied to resource.
```

Mode comparison:

```text
base:
ood_accuracy = 1.000
spurious_attack_accuracy = 1.000
relation_subspace_drop = 0.483
nuisance_subspace_drop = 0.000
gated_neural_relation_score = 0.819

shortcut:
ood_accuracy = 0.569
spurious_attack_accuracy = 0.111
relation_subspace_drop = 0.441
nuisance_subspace_drop = 0.441
gated_neural_relation_score = 0.000
```

Interpretation:

```text
base uses relation-specific structure;
shortcut has readable relation information but does not use it cleanly.
```

## 4.6 Neural-to-Table Bridge

The third step extracts an explicit editable relation table from the neural model and probe.

Key code:

```python
@dataclass
class ExtractedRule:
    condition: dict[str, str]
    outcome: str
    confidence: float
    probe_relation: str
```

Table interface:

```python
class ExtractedRelationTable:
    def infer_resource(self, context: dict[str, str]) -> str:
        for rule in self.rules:
            if all(context[key] == value for key, value in rule.condition.items()):
                return rule.outcome
        return "neutral"

    def act(self, context: dict[str, str]) -> str:
        return "eat" if self.infer_resource(context) == "food" else "avoid"

    def edit_rule(self, condition: dict[str, str], new_outcome: str) -> bool:
        for rule in self.rules:
            if rule.condition == condition:
                rule.outcome = new_outcome
                return True
        ...
```

Extraction:

```python
def extract_relation_table(model: MLPClassifier, relation_probe) -> ExtractedRelationTable:
    rules = []
    for texture in TEXTURES:
        for wet in WETS:
            context = canonical_context(texture, wet)
            x = encode_context(context).reshape(1, -1)
            hidden = hidden_states(model, x)
            relation_probs = relation_probe.predict_proba(hidden)[0]
            relation_index = int(np.argmax(relation_probs))
            resource_index = int(model.predict(x)[0])
            rules.append(
                ExtractedRule(
                    condition={"texture": texture, "wet": wet},
                    outcome=RESOURCES[resource_index],
                    confidence=float(relation_probs[relation_index]),
                    probe_relation=RELATIONS[relation_index],
                )
            )
    return ExtractedRelationTable(rules)
```

Result:

```text
base:
table_ood_accuracy = 1.000
table_spurious_attack_accuracy = 1.000
table_relation_alignment = 1.000
gated_extraction_score = 1.000

shortcut:
table_ood_accuracy ~= 0.455
table_spurious_attack_accuracy ~= 0.456
table_relation_alignment ~= 0.333
gated_extraction_score = 0.000
```

Meaning:

```text
neural representation -> explicit relation table -> editable action policy
```

works only when the neural model has a stable relation representation.

---

## 5. Project 3: `slope-relation-toy`

## 5.1 Purpose

This project moves the framework into a slope-engineering review toy world.

It is not a real slope simulation.

It is a diagnostic harness for:

```text
Does the AI bind recommendations to an engineering relation chain?
```

## 5.2 Core Relation Chain

```text
Rainfall -> Infiltration
Infiltration -> PorePressure
PorePressure -> Displacement
Displacement -> CrackExpansion
CrackExpansion -> RiskUp

Drainage -> PorePressureDown
Anchoring -> DisplacementDown
ToeExcavation -> StabilityDown
Monitoring -> UncertaintyDown
StopWork -> ExposureRiskDown
```

## 5.3 Environment Code

Key chain evaluation:

```python
def evaluate_chain(
    context: dict[str, str],
    drainage_effective: bool = True,
    anchoring_effective: bool = True,
) -> ChainState:
    infiltration_high = context["rainfall"] == "high"
    pore_high = infiltration_high
    if drainage_effective and context["drainage"] == "good":
        pore_high = False

    displacement_high = pore_high or context["toe_excavation"] == "yes"
    if anchoring_effective and context["anchoring"] == "present":
        displacement_high = False

    crack_yes = displacement_high and context["monitoring"] == "sparse"
    if crack_yes:
        risk = "high"
    elif pore_high or displacement_high:
        risk = "medium"
    else:
        risk = "low"

    if risk == "high":
        action = "stop_work"
    elif pore_high:
        action = "drain"
    elif displacement_high:
        action = "anchor"
    else:
        action = "monitor"

    return ChainState(...)
```

Reward:

```python
def reward_for(action: str, optimal_action: str, risk: str) -> float:
    if action == optimal_action:
        return 1.0
    if risk == "high" and action == "monitor":
        return -1.0
    if risk == "high":
        return -0.5
    if risk == "medium" and action == "monitor":
        return -0.2
    return 0.0
```

## 5.4 Agents

Implemented agents:

```text
majority
surface
structural_memory
generic_review
relation_chain
learned_links
```

### `surface`

Learns warning labels:

```text
weather_label, contractor_report -> action
```

It performs well in training but fails OOD and spurious attack.

### `structural_memory`

Memorizes structural context:

```python
def _key(self, context):
    return (
        context["rainfall"],
        context["drainage"],
        context["anchoring"],
        context["toe_excavation"],
        context["monitoring"],
    )
```

This can predict very well, including OOD and spurious attack, but has no:

```text
editable relation chain
relation audit
engineering review chain
```

### `generic_review`

Produces plausible engineering words:

```python
return {
    "variables": ["rainfall", "slope risk"],
    "relation_chain": ["rainfall affects slope safety", "support improves stability"],
    "action": "drain",
    "action_point": "strengthen drainage and support",
    ...
}
```

It is intentionally rejected because it does not bind actions to concrete relation-chain intervention points.

### `relation_chain`

Oracle-style hand-written relation-chain model.

Key act logic:

```python
def act(self, context):
    state = evaluate_chain(
        context,
        drainage_effective=self.links["Drainage -> PorePressureDown"],
        anchoring_effective=self.links["Anchoring -> DisplacementDown"],
    )
    return state.optimal_action
```

Editable links:

```python
self.links = {
    "Drainage -> PorePressureDown": True,
    "Anchoring -> DisplacementDown": True,
}
```

### `learned_links`

This is the first non-oracle relation-chain baseline.

It starts with key links disabled:

```python
self.links = {
    "Drainage -> PorePressureDown": False,
    "Anchoring -> DisplacementDown": False,
    "WeatherLabel -> RiskUp": False,
    "ContractorReport -> RiskUp": False,
    "Drainage -> CrackDown": False,
}
```

It observes training samples and learns whether two predefined candidate links are effective:

```python
def _refresh_links(self) -> None:
    drain_support = sum(self.drainage_good_high_rain.values())
    if drain_support >= self.min_support:
        self.links["Drainage -> PorePressureDown"] = (
            self.drainage_good_high_rain["monitor"]
            >= self.drainage_good_high_rain["drain"]
        )

    anchor_support = sum(self.anchored_excavation.values())
    if anchor_support >= self.min_support:
        self.links["Anchoring -> DisplacementDown"] = (
            self.anchored_excavation["monitor"]
            >= self.anchored_excavation["anchor"]
        )
```

It also keeps irrelevant distractor links disabled:

```text
WeatherLabel -> RiskUp
ContractorReport -> RiskUp
Drainage -> CrackDown
```

Important boundary:

```text
learned_links learns whether predefined candidate links are effective.
It does not discover an unrestricted engineering relation graph.
```

---

## 6. Engineering Review Layer

The slope toy includes an AI plan-review template.

Review items:

```text
1. Identifies key variables
2. Explains variable relation chains
3. Explains intervention action points
4. Defines verification indicators
5. Defines failure conditions
6. Defines human takeover conditions
7. Defines responsibility chain
```

Scoring:

```text
0 = not shown
1 = partially shown
2 = clearly shown
```

Total:

```text
14 points
```

Key scoring code:

```python
def score_item(plan: dict[str, object], key: str) -> int:
    value = plan.get(key)
    if value is None or value == "" or value == []:
        return 0
    if key == "relation_chain":
        if not isinstance(value, list):
            return 0
        linked = [item for item in value if isinstance(item, str) and "->" in item]
        return 2 if len(linked) >= 3 else int(len(linked) > 0)
    if key == "action_point":
        if not isinstance(value, str):
            return 0
        return 2 if "->" in value else 1
    if isinstance(value, list):
        return 2 if len(value) >= 2 else 1
    return 2
```

Plan scoring:

```python
def score_review_plan(plan: dict[str, object]) -> dict[str, object]:
    item_scores = {key: score_item(plan, key) for key in REVIEW_ITEMS}
    total = sum(item_scores.values())
    return {
        "item_scores": item_scores,
        "total": total,
        "max_total": 14,
        "passed": total >= 10,
    }
```

## 6.1 Review Consistency

The system must not only fill review fields.

It must bind each scenario to the correct action point:

```text
drain      -> Drainage -> PorePressureDown
anchor     -> Anchoring -> DisplacementDown
stop_work  -> StopWork -> ExposureRiskDown
monitor    -> Monitoring -> UncertaintyDown
```

Key code:

```python
def review_consistency(agent: BaseAgent) -> float:
    cases = [
        (..., "drain", "Drainage -> PorePressureDown"),
        (..., "stop_work", "StopWork -> ExposureRiskDown"),
        (..., "anchor", "Anchoring -> DisplacementDown"),
        (..., "monitor", "Monitoring -> UncertaintyDown"),
    ]
    passed = 0
    for context, expected_action, expected_point in cases:
        plan = agent.review_plan(context)
        passed += int(
            plan.get("action") == expected_action
            and plan.get("action_point") == expected_point
        )
    return passed / len(cases)
```

---

## 7. Slope Gated Score

The slope toy uses a strict gated score.

Current gates:

```text
ood_success >= 0.8
spurious_attack_success >= 0.8
counterfactual_accuracy >= 0.8
edit_success >= 1.0
relation_audit >= 0.9
irrelevant_link_rejection >= 0.9
noisy_observation_success >= 0.7
review_score >= 0.9
review_consistency >= 0.9
```

Key code:

```python
def gated_score(metrics: dict[str, float]) -> float:
    gates = [
        metrics["ood_success"] >= 0.8,
        metrics["spurious_attack_success"] >= 0.8,
        metrics["counterfactual_accuracy"] >= 0.8,
        metrics["edit_success"] >= 1.0,
        metrics["relation_audit"] >= 0.9,
        metrics["irrelevant_link_rejection"] >= 0.9,
        metrics["noisy_observation_success"] >= 0.7,
        metrics["review_score"] >= 0.9,
        metrics["review_consistency"] >= 0.9,
    ]
    if not all(gates):
        return 0.0
    return (
        0.2 * metrics["ood_success"]
        + 0.2 * metrics["spurious_attack_success"]
        + 0.16 * metrics["counterfactual_accuracy"]
        + 0.14 * metrics["edit_success"]
        + 0.1 * metrics["relation_audit"]
        + 0.05 * metrics["irrelevant_link_rejection"]
        + 0.08 * metrics["noisy_observation_success"]
        + 0.04 * metrics["review_score"]
        + 0.03 * metrics["review_consistency"]
    )
```

---

## 8. Noisy Monitoring Test

A minimal monitoring-noise test was added.

The true state is evaluated from the true context.

The agent sees a corrupted observation:

```python
def corrupt_monitoring_observation(
    context: dict[str, str],
    rng: np.random.Generator,
    noise_rate: float = 0.1,
) -> dict[str, str]:
    observed = dict(context)
    if rng.random() < noise_rate:
        observed["monitoring"] = (
            "dense" if context["monitoring"] == "sparse" else "sparse"
        )
    return observed
```

Evaluation:

```python
def noisy_observation_success(agent, seed, n_steps=200, noise_rate=0.1) -> float:
    env = SlopeToyWorld(seed=seed, mode="ood")
    rng = np.random.default_rng(seed + 1000)
    successes = []
    for _ in range(n_steps):
        true_context = env.sample_context()
        observed_context = corrupt_monitoring_observation(true_context, rng, noise_rate)
        true_state = evaluate_chain(true_context)
        successes.append(agent.act(observed_context) == true_state.optimal_action)
    return sum(int(success) for success in successes) / len(successes)
```

Current interpretation:

```text
This is not a calibrated monitoring-noise safety test.
It is only a basic robustness diagnostic.
```

---

## 9. Current Slope Results

Mean over seeds:

```text
agent              ood   spurious  cf     edit  audit  irrelevant  noisy  review  consistency  gated
learned_links     1.000 1.000     1.000  1.0   1.0    1.0         0.753  1.0     1.0          0.980
relation_chain    1.000 1.000     1.000  1.0   1.0    1.0         0.753  1.0     1.0          0.980
structural_memory 1.000 1.000     1.000  0.0   0.0    1.0         0.976  0.0     0.0          0.000
surface           0.327 0.118     0.333  0.0   0.0    1.0         0.312  0.0     0.0          0.000
generic_review    0.193 0.162     0.333  0.0   0.0    1.0         0.192  0.429   0.0          0.000
majority          0.571 0.581     0.667  0.0   0.0    1.0         0.570  0.0     0.0          0.000
```

Most important comparison:

```text
structural_memory predicts well, including OOD and spurious attack,
but fails because it has no editable/auditable/reviewable relation chain.
```

This is central to the project.

It shows that the gates do not reduce relation internalization to predictive accuracy.

---

## 10. Self-Audit

The project contains a self-audit report:

```text
F:\slope-relation-toy\reports\self_audit.md
```

The self-audit states:

```text
The project is useful as a relation-chain diagnostic,
but it is not evidence of a deployable engineering AI system.
```

Known weaknesses:

```text
relation_chain is oracle-style.
learned_links only learns predefined candidate links.
toy world is deterministic.
thresholds are manually chosen.
monitoring noise is single-step corruption.
there is no time dynamics.
there is no numerical geotechnical model.
```

False-positive risks:

```text
a model could memorize fixed review-consistency cases.
a model could pass by learning the small candidate-link set.
a template could encode the same action-point map.
```

---

## 11. What Can Be Claimed

Supported claims:

```text
1. A runnable relation-internalization diagnostic framework has been implemented.
2. The framework distinguishes prediction, memory, surface cues, generic review text, and editable relation chains.
3. Neural relation probes must be paired with causal intervention tests.
4. Neural hidden representations can be converted into explicit editable relation tables in the toy setting.
5. The slope toy shows how engineering review can be tied to relation-chain action points.
6. A minimal predefined-link learner can pass the same slope gates after observing enough samples.
```

Unsupported claims:

```text
1. Real slope engineering safety capability.
2. Autonomous discovery of arbitrary engineering relations.
3. Calibration of safety thresholds.
4. Deployment readiness.
5. General proof of relation internalization in large AI systems.
```

---

## 12. Recommended Next Step

Do not add more isolated metrics immediately.

The best next step is:

```text
time-dependent monitoring / delayed effects
```

Minimal version:

```text
rainfall at t
-> pore pressure at t+1
-> displacement at t+2
-> crack/risk at t+3
```

Why this matters:

```text
Real engineering relations are delayed.
If the model only handles same-step toy relations, it may not be internalizing process structure.
```

Next tests should ask:

```text
Can learned_links infer delayed cause-effect links?
Can it distinguish immediate surface warning from delayed physical response?
Can it edit a delayed link and update future action policy?
```

---

## 13. Current Verification Status

```text
F:\relation-internalization-test: 17 passed
F:\neural-relation-probe: 7 passed
F:\slope-relation-toy: 12 passed
```

---

## 14. V2: Temporal Slope Relation Toy

V2 has now been implemented at:

```text
F:\temporal-slope-relation-toy
```

V2 does not add more static review metrics. It tests whether the relation enters time:

```text
rainfall[t] -> pore_pressure[t+1]
pore_pressure[t] -> displacement[t+1]
displacement[t] + sparse monitoring[t] -> crack/risk[t+1]
```

### 14.1 V2 Agents

```text
surface_temporal
structural_memory_temporal
instant_relation_chain
delayed_relation_chain
learned_delayed_links
```

The key comparison is:

```text
structural_memory_temporal can predict the sequence well,
but it cannot edit delay and cannot audit temporal relations.
```

### 14.2 Key Code

Temporal chain evaluation:

```python
def evaluate_temporal_chain(seq: list[TemporalStep], rainfall_delay: int = 1) -> list[TemporalStep]:
    for t, step in enumerate(seq):
        rain_t = t - rainfall_delay
        pore_high = rain_t >= 0 and seq[rain_t].rainfall == "high"
        if t - 1 >= 0 and seq[t - 1].drainage == "good":
            pore_high = False
        step.pore_pressure = "high" if pore_high else "low"

        displacement_high = (t - 1 >= 0 and seq[t - 1].pore_pressure == "high") or step.toe_excavation == "yes"
        if t - 1 >= 0 and seq[t - 1].anchoring == "present":
            displacement_high = False
        step.displacement = "high" if displacement_high else "low"

        crack_yes = t - 1 >= 0 and seq[t - 1].displacement == "high" and seq[t - 1].monitoring == "sparse"
        step.crack = "yes" if crack_yes else "no"
```

Learned delayed-link selection:

```python
class LearnedDelayedLinksAgent(DelayedRelationChainAgent):
    def observe_sequence(self, seq: list[TemporalStep]) -> None:
        for delay in [1, 2]:
            matches = 0
            total = 0
            for t in range(delay, len(seq)):
                total += 1
                predicted_high = seq[t - delay].rainfall == "high" and seq[t - 1].drainage != "good"
                matches += int((seq[t].pore_pressure == "high") == predicted_high)
            self.delay_votes[delay] += matches - (total - matches)
        if self.delay_votes:
            self.rainfall_delay = self.delay_votes.most_common(1)[0][0]
```

Delay-edit gate:

```python
def delay_edit_success(agent: BaseTemporalAgent) -> float:
    before_t1 = agent.act(seq, 1)
    before_t2 = agent.act(seq, 2)
    if not agent.edit_delay("Rainfall -> PorePressure", 2):
        return 0.0
    after_t1 = agent.act(seq, 1)
    after_t2 = agent.act(seq, 2)
    return float(before_t1 == "drain" and before_t2 != "drain" and after_t1 != "drain" and after_t2 == "drain")
```

### 14.3 V2 Gates

```text
temporal_ood_success >= 0.8
delayed_counterfactual_accuracy >= 0.8
delay_edit_success >= 0.9
surface_shortcut_rejection >= 0.8
temporal_audit_score >= 0.9
```

### 14.4 V2 Results

Mean over seeds 0-4:

```text
agent                       ood    cf   edit  shortcut  audit  gated
delayed_relation_chain      1.000  1.0  1.0   1.000     1.0    1.0
learned_delayed_links       1.000  1.0  1.0   1.000     1.0    1.0
structural_memory_temporal  1.000  1.0  0.0   0.967     0.0    0.0
instant_relation_chain      0.412  0.0  0.0   0.369     0.0    0.0
surface_temporal            0.264  0.0  0.0   0.261     0.0    0.0
```

### 14.5 V2 Claim Boundary

Supported:

```text
The toy distinguishes delayed relation-chain use from surface temporal shortcuts, same-step relation logic, and temporal state memory.
```

Not supported:

```text
real geotechnical time-series modeling
unrestricted temporal relation discovery
calibrated safety thresholds
deployment as engineering review AI
```

Updated verification:

```text
F:\temporal-slope-relation-toy: 5 passed
F:\slope-relation-toy: 12 passed
F:\relation-internalization-test: 17 passed
F:\neural-relation-probe: 7 passed
```

### 15.9 Freeze Artifacts

V2.1 is frozen with:

```text
F:\temporal-slope-relation-toy\reports\V2_1_FINAL_REPORT.md
F:\temporal-slope-relation-toy\reports\V2_1_LIMITATIONS.md
F:\temporal-slope-relation-toy\reports\V2_1_CLAIMS.md
```

The next stage has a charter:

```text
F:\temporal-slope-relation-toy\reports\V3_CHARTER.md
```

V3 target:

```text
partial observability + missing sensors + delayed noisy observations + takeover threshold
```

V3 should test:

```text
Temporal relation internalization is not enough; engineering-grade relation use requires uncertainty-aware takeover.
```

---

## 15. V2.1 Reviewer Hardening

V2.1 has now been implemented at:

```text
F:\temporal-slope-relation-toy
```

It does not change the main V2 claim. It hardens the V2 temporal toy against reviewer concerns:

```text
fixed templates
limited predefined delay candidates
surface temporal shortcuts
single-link-only editing
audit strings without real time indexes
```

### 15.1 New Hardening Attacks

```text
1. variable delay
2. false temporal shortcut
3. multi-link delay edit
4. temporal audit consistency
5. anti-template generalization
```

### 15.2 Updated Delay Map

V2.1 generalizes from one rainfall delay to a multi-link delay map:

```python
def default_delay_map(rainfall_delay: int = 1) -> dict[str, int]:
    return {
        "Rainfall -> PorePressure": rainfall_delay,
        "PorePressure -> Displacement": 1,
        "Displacement -> Crack": 1,
        "Drainage -> PorePressureDown": 1,
        "Anchoring -> DisplacementDown": 1,
    }
```

The temporal chain now evaluates delayed interventions with:

```text
Rainfall[t-d1] high -> PorePressure[t] high
Drainage[t-dD] good -> PorePressure[t] low
PorePressure[t-d2] high -> Displacement[t] high
Anchoring[t-dA] present -> Displacement[t] low
Displacement[t-d3] high + Monitoring[t-d3] sparse -> Crack[t]
```

### 15.3 Learned Multi-Link Delays

`learned_delayed_links` now tracks delay votes per candidate link:

```python
self.candidate_delays = [1, 2, 3]
self.candidate_links = [
    "Rainfall -> PorePressure",
    "PorePressure -> Displacement",
    "Displacement -> Crack",
    "Drainage -> PorePressureDown",
    "Anchoring -> DisplacementDown",
]
self.delay_votes: dict[str, Counter[int]]
self.delay_map: dict[str, int]
```

For each sequence, it scores link-delay agreement and refreshes the delay map:

```python
def observe_sequence(self, seq: list[TemporalStep]) -> None:
    for link in self.candidate_links:
        for delay in self.candidate_delays:
            self.delay_votes[link][delay] += self._score_link_delay(seq, link, delay)
    self._refresh_delay_map()
```

### 15.4 Time-Indexed Audit

V2.1 requires audit strings with explicit time indexes:

```python
def temporal_audit(self) -> list[str]:
    return [
        f"Rainfall[t-{d1}] -> PorePressure[t]",
        f"Drainage[t-{d_drain}] -> PorePressureDown[t]",
        f"PorePressure[t-{d2}] -> Displacement[t]",
        f"Anchoring[t-{d_anchor}] -> DisplacementDown[t]",
        f"Displacement[t-{d3}] + Monitoring[t-{d3}] -> Crack[t]",
    ]
```

Audit strings like:

```text
Rainfall -> PorePressure
```

do not pass V2.1.

### 15.5 Hardening Gates

```text
variable_delay_success >= 0.8
false_delay_shortcut_rejection >= 0.8
multi_link_delay_edit_success >= 0.8
temporal_audit_consistency >= 0.9
anti_template_generalization >= 0.8
```

If any gate fails:

```text
hardening_gated_score = 0.0
```

### 15.6 V2.1 Results

Mean over seeds 0-4:

```text
agent                       variable  shortcut  edit  audit  generalization  hardening
delayed_relation_chain      1.000     1.000     1.0   1.0    1.000           1.0
learned_delayed_links       1.000     1.000     1.0   1.0    1.000           1.0
structural_memory_temporal  0.985     0.980     0.0   0.0    0.975           0.0
instant_relation_chain      0.426     0.380     0.0   0.0    0.383           0.0
surface_temporal            0.296     0.291     0.0   0.0    0.291           0.0
```

### 15.7 Interpretation

V2.1 strengthens the central V2 conclusion:

```text
temporal prediction is not temporal relation internalization.
```

The strongest negative control is still:

```text
structural_memory_temporal
```

It performs well on variable delay and shortcut rejection:

```text
variable_delay_success = 0.985
false_delay_shortcut_rejection = 0.980
anti_template_generalization = 0.975
```

but fails:

```text
multi_link_delay_edit_success = 0.0
temporal_audit_consistency = 0.0
hardening_gated_score = 0.0
```

This shows that V2.1 is not merely rewarding temporal prediction.

### 15.8 V2.1 Claim Boundary

Supported:

```text
The delayed relation-chain diagnostic is hardened against fixed-template shortcuts, false temporal shortcuts, and single-link-only edit tests.
```

Not supported:

```text
real geotechnical time-series modeling
unrestricted temporal relation discovery
deployment-ready engineering safety AI
```

Updated verification:

```text
F:\temporal-slope-relation-toy: 10 passed
F:\slope-relation-toy: 12 passed
F:\relation-internalization-test: 17 passed
F:\neural-relation-probe: 7 passed
```
