# R4 Structural Internalization Substrate Experiment Spec

Date: 2026-04-29

## 1. Core Question

R4 asks a different question from the earlier gates.

Not:

```text
Can more training or better prompting make a system pass?
```

But:

```text
What kind of internal substrate is eligible to carry structural relation
internalization?
```

In the egg/stone analogy:

```text
An egg can hatch because it already has the right generative substrate.
A stone cannot hatch by being warmed longer.
Rice can become cooked rice; sand cannot become food by longer boiling.
```

R4 should test whether a system's internal substrate is capable of becoming a
usable relation structure under pressure.

## 2. Working Definition

A substrate is eligible for structural relation internalization if it can support
all of the following:

```text
factorization: relation parts can be separated rather than stored as one blob
binding: variables, values, support/context, and links can be bound together
local update: one relation can change without rewriting unrelated behavior
composition: learned links can be chained for counterfactuals and action
audit access: uncertainty can be assigned to specific links or chain nodes
inspection value: missing observations can be ranked by relation relevance
```

The substrate does not need to be symbolic. But if it cannot expose these
properties behaviorally, it should not be counted as structurally internalized.

## 3. Experimental Principle

Use the same world, data budget, and training pressure across different
substrates.

The independent variable is the substrate:

```text
What internal form and update mechanism does the system have?
```

The dependent variable is gated structural behavior:

```text
Does it transfer, counterfactually simulate, edit locally, audit exactly, and
select inspections under budget?
```

## 4. Substrate Ladder

### S0: Lookup / Memory Substrate

Role:

```text
stone baseline
```

Internal form:

```text
observed state/action -> remembered best response
```

Expected behavior:

```text
May fit seen states.
Should fail OOD transfer, local edits, exact audit, and inspect selection.
```

Purpose:

```text
Shows that successful behavior on seen states is not enough.
```

### S1: Flat Predictor Substrate

Role:

```text
smooth stone baseline
```

Internal form:

```text
single vector or flat network mapping observations to outcomes/actions
```

Training pressure:

```text
prediction and action reward only
```

Expected behavior:

```text
May generalize statistically.
Should fail local relation edit and exact audit unless the architecture exposes
relation structure.
```

Purpose:

```text
Tests whether compression alone is enough. It should not be.
```

### S2: Flat Predictor With Edit Pressure

Role:

```text
heated stone test
```

Internal form:

```text
same flat substrate as S1
```

Training pressure:

```text
prediction + counterfactual + edit examples
```

Expected behavior:

```text
May learn to imitate edit compliance.
Should still fail support-conditioned local edit and exact audit if it has no
factorized relation carrier.
```

Purpose:

```text
Tests whether pressure alone can create structure without the right substrate.
```

### S3: Relation Slot Substrate

Role:

```text
candidate egg
```

Internal form:

```text
slots for candidate links, e.g. cause slot, effect slot, support/context slot,
confidence, last-update trace
```

Training pressure:

```text
transition prediction + counterfactual + local edit + audit + inspect value
```

Expected behavior:

```text
Should be able to pass if the slot update and retrieval mechanisms are adequate.
```

Purpose:

```text
Tests whether explicit factorization is sufficient.
```

### S4: Differentiable Graph Substrate

Role:

```text
neural egg candidate
```

Internal form:

```text
learned edge weights or relation embeddings over variables; message passing or
structural attention composes links.
```

Training pressure:

```text
prediction + interventions + counterfactuals + edit locality + audit target +
budgeted inspect
```

Expected behavior:

```text
Should pass only if relation edges are modular enough for local edits and audit.
```

Purpose:

```text
Tests whether a neural substrate can carry the same structural behavior without
a hand-written relation table.
```

### S5: Black-Box LLM Prompt Substrate

Role:

```text
frozen negative sidecar
```

Internal form:

```text
unknown black-box language model state
```

Expected behavior from current evidence:

```text
Can answer some simple relation prompts; fails full budgeted inspect,
behavior-level local edit, and exact audit gates.
```

Purpose:

```text
Not a constructive route. Keep as comparison only.
```

## 5. Core Tests

Each substrate must run through the same R4 gates.

### Gate 1: Transfer

Same relation, new symbols/support.

Pass condition:

```text
correct target behavior under randomized variable names and support contexts
```

### Gate 2: Counterfactual Composition

Intervene on an intermediate variable and predict downstream effects.

Pass condition:

```text
correct downstream change without changing unrelated variables
```

### Gate 3: Local Edit Behavior

Edit one relation in one support/context.

Pass condition:

```text
target query changes
other support unchanged
unrelated chain unchanged
```

### Gate 4: Exact Audit

Hide or corrupt one relation-chain node.

Pass condition:

```text
inspect target is correct
uncertainty is correct
audit link is exact and directional
irrelevant missingness produces no audit
```

### Gate 5: Budgeted Inspection

Multiple variables may be missing, inspection budget is limited.

Pass condition:

```text
selected inspection maximizes relation-relevant decision value
```

### Gate 6: Structural Compression

Train on fewer contexts than the combinatorial state space.

Pass condition:

```text
generalizes by relation composition rather than memorizing all state/action pairs
```

### Gate 7: Substrate Intervention

Intervene directly on the substrate representation.

Examples:

```text
delete one edge
swap one support binding
freeze one relation slot
zero one relation embedding
```

Pass condition:

```text
behavior changes only where that structural part is used
```

This is the strongest substrate test. It asks whether the internal carrier is
causally responsible, not just correlated with behavior.

## 6. Key Metrics

```text
transfer_accuracy
counterfactual_accuracy
local_edit_target_success
local_edit_non_target_stability
exact_audit_score
budgeted_inspect_accuracy
inspection_value_efficiency
structural_compression_ratio
substrate_intervention_locality
gated_substrate_score
```

The gated score is zero unless all required gates pass.

## 7. Necessary Controls

False positives to include:

```text
memory_table
flat_predictor
flat_predictor_with_edit_examples
global_mapping
edit_acknowledgement
first_missing_inspect
always_inspect
generic_audit
reverse_audit_link
outcome_default_audit
```

The point is not to beat weak baselines only. The important baseline is:

```text
a system with enough capacity to fit behavior but no eligible structural
carrier.
```

## 8. Minimal First Implementation

R4.0 should start small:

```text
world: reuse relation-agent-r1 process world
seeds: 5
substrates: S0 memory, S1 flat predictor, S3 relation slots, S4 small graph
gates: transfer, local edit behavior, exact audit, budgeted inspect
```

Leave stochastic dynamics and richer domains for later.

## 9. Interpretation Rules

Allowed claim if a substrate passes:

```text
This substrate is eligible for structural relation internalization in the toy
setting.
```

Not allowed:

```text
This proves general intelligence.
This proves real engineering understanding.
This proves all neural models can internalize relations.
```

If all neural substrates fail:

```text
The tested neural substrates did not carry structural relation internalization
under the tested pressure.
```

Do not conclude:

```text
Neural systems cannot internalize relations.
```

## 10. Next Work Items

```text
1. Implement R4.0 as a new stage, not as more LLM prompts.
2. Reuse R3 active-inspection cases where possible.
3. Add substrate-specific intervention APIs.
4. Compare flat predictor vs relation-slot vs differentiable graph.
5. Freeze the result with report, self-audit, and claim boundary.
```
