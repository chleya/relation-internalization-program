# Project Evolution

Date: 2026-04-28

Source discussion:

```text
https://chatgpt.com/share/69f0a27e-e944-839a-a3ca-a7a7f9023e02
```

This file records how the current mainline emerged from a broader discussion. It is not evidence by itself. The runnable projects, tests, metrics, reports, and negative controls are the evidence.

## 1. Starting Question

The starting point was not relation internalization. The first question was broader:

```text
AI can generate complex systems faster than humans can understand, verify, maintain, and take responsibility for them.
```

The early concern was not ordinary job displacement. It was a deeper mismatch:

```text
complexity production rate > human cognitive absorption rate
```

This led to the idea of a "cognitive debt" model: AI-generated complexity may accumulate faster than human institutions, engineers, auditors, and operators can absorb it.

## 2. First Formal Direction: Cognitive Debt

The discussion then tried to formalize this as:

```text
AI complexity production
vs
human cognitive carrying capacity
```

The useful part of this phase was the shift from a vague social concern to measurable failure modes:

```text
understanding gap
verification gap
maintenance gap
responsibility gap
takeover gap
```

However, this direction was too broad to implement directly. It touched engineering safety, labor structure, education, governance, critical infrastructure, and AI risk at the same time.

The project needed a narrower test object.

## 3. Key Narrowing

The central question became:

```text
If AI produces complex decisions or designs, what must be internally available for humans to verify, edit, audit, and take over?
```

This narrowed the problem from:

```text
Can AI generate complexity?
```

to:

```text
Does the system expose usable internal relations?
```

That narrowing created the current main claim:

```text
Relation-internalized systems should expose usable internal relations.
```

"Usable" means the relation can support:

```text
transfer
counterfactual intervention
internal editing
auditing
action guidance
eventual takeover
```

## 4. Why Not Start With Real Engineering?

The discussion repeatedly returned to engineering examples such as aerospace, infrastructure, and slope safety. Those examples motivated the project, but they were not suitable as the first experimental target.

Reasons:

```text
real data is noisy and sparse
engineering safety claims are high-stakes
geotechnical mechanisms require domain validation
responsibility and deployment boundaries are strict
real projects would obscure the core theoretical test
```

Therefore the first stage had to be toy diagnostics.

The guiding rule became:

```text
Use engineering examples to motivate the question, but do not claim real engineering capability.
```

## 5. V1: Static Relation Internalization

The first runnable project became:

```text
relation-internalization-test
```

It used a simple food/poison world to distinguish:

```text
majority action
context memory
black-box fitting
prediction
explicit relation internalization
```

The important shift was that success required more than in-distribution reward. The model had to support:

```text
OOD transfer
relation reversal
counterfactual response
internal edit
edit locality
```

This produced the first operational version of the claim:

```text
high prediction performance is not enough.
```

## 6. Neural Evidence Chain

The next concern was:

```text
Could a neural model contain relation information internally even if it does not expose a symbolic relation table?
```

This led to:

```text
neural-relation-probe
```

The project added:

```text
hidden-state probes
relation and nuisance subspace tests
causal subspace removal
neural-to-table extraction
```

The key conclusion was intentionally cautious:

```text
probe-readable is not the same as causally used.
```

This preserved the main standard:

```text
relation internalization requires usable relations, not merely readable latent signals.
```

## 7. Engineering Toy: Slope Relation Chain Review

The next step was to move from food-world relations to a slope-engineering-style toy world:

```text
slope-relation-toy
```

The purpose was not real geotechnical modeling. It was to test whether an AI review policy could bind actions to relation chains:

```text
rainfall -> infiltration -> pore pressure -> displacement -> crack/risk
drainage -> pore pressure down
anchoring -> displacement down
monitoring -> uncertainty down
```

This stage separated:

```text
generic review language
surface labels
structural memory
relation-chain action review
```

The important negative control was:

```text
structural memory can score high on prediction but fail edit, audit, and reviewability.
```

## 8. V2: Delayed Temporal Relation Chains

The discussion then identified a weakness in the static toy setup:

```text
real engineering relations are delayed and process-like, not same-step mappings.
```

This led to:

```text
temporal-slope-relation-toy
```

V2 changed the relation from:

```text
Rainfall -> PorePressure -> Displacement -> Risk
```

to:

```text
rainfall[t] -> pore_pressure[t+1]
pore_pressure[t] -> displacement[t+1]
displacement[t] -> crack/risk[t+1]
```

The core claim became:

```text
temporal prediction is not temporal relation internalization.
```

## 9. V2.1: Reviewer Hardening

After V2, the likely reviewer objection was:

```text
learned_delayed_links may pass only because the test uses fixed templates, fixed delays, and limited candidate links.
```

V2.1 was added to attack that possibility:

```text
variable delay
false temporal shortcut
multi-link delay edit
temporal audit consistency
anti-template generalization
```

The strongest result was the contrast:

```text
structural_memory_temporal:
  high variable_delay_success
  high false_delay_shortcut_rejection
  high anti_template_generalization
  zero multi_link_delay_edit_success
  zero temporal_audit_consistency
  zero hardening_gated_score
```

This sharpened the distinction:

```text
temporal sequence prediction can be strong while temporal relation internalization is absent.
```

## 10. Why V3 Was Needed And Why V4 Is Next

V1, neural probe, slope toy, V2, and V2.1 all point to the same next weakness:

```text
relations are not enough if observations are missing, delayed, noisy, or conflicting.
```

Therefore the next mainline is:

```text
V3: Partial Observability + Missing Sensors + Delayed Noisy Observations + Takeover Threshold
```

V3 should test whether a system can:

```text
use relation chains under uncertainty
identify uncertain links
avoid unsafe automation
trigger takeover
explain why takeover is needed
```

The next claim should not be:

```text
the system is engineering-safe
```

It should be:

```text
engineering-relevant relation use requires uncertainty-aware takeover.
```

V3, V3.1, and V3.2 then tested:

```text
missing sensors
delayed noisy observations
unsafe automation
relation-specific takeover
overcautious takeover false positives
long missing spans
correlated sensor failures
drift-like conflicts
multi-conflict audit
```

After V3.2, the next weakness is no longer another toy uncertainty metric.
The next weakness is workflow expression:

```text
Can the relation-chain and takeover diagnostic be expressed as a bounded
engineering-review case without making real safety claims?
```

That is the V4 direction:

```text
V4: Small Engineering Review Case
```

## 11. What Was Excluded

Several directions were discussed but deliberately excluded from the mainline.

### Broad AI Society Theory

Useful as motivation, but too broad for the current research program.

### Real Slope Engineering Data

Deferred because the current project is a diagnostic toy program, not a safety-calibrated geotechnical model.

### General Agent Governance

Relevant later through CEE, unified-sel, and highway-agent-kernel, but not evidence for relation internalization.

### Symbol Emergence and Continuous Communication

Projects such as `symbologenesis-boundary` and `cep-cc` remain independent research lines.

### Product Integration

Projects such as `NeuralSite-Godot` may later host workflows, but should not be used as research evidence now.

## 12. Current Mainline Statement

The project should now be described as:

```text
Relation Internalization Program:
a runnable diagnostic chain testing whether systems expose usable internal relations,
distinguished from prediction, memory, shortcut use, generic review text, and probe readability.
```

Current supported ladder:

```text
V1: static relation internalization
V1.5: neural readability vs causal use
slope toy: engineering relation-chain review
V2: delayed temporal relation chain
V2.1: reviewer hardening against template and shortcut false positives
V3: uncertainty-aware takeover
V3.1: takeover hardening against overcautious templates
V3.2: partial-observability stress tests
V4: bounded engineering-review-case diagnostic
V4.1: review hardening against schema/prose/order/boilerplate/approval false positives
V4.2: adversarial case mutation against field order, irrelevant variables, paraphrases, and hidden unsafe phrases
V5: governance shell for logs, approval gates, replay, and responsibility traces
V5.1: governance hardening against fake replay, gate-label-only, responsibility boilerplate, missing evidence, and route tampering
V6: multi-party audit resolution for conflicting review outputs
```

## 13. Correction After V6: Return To Agent Internals

After V6, we reassessed the route against the original question:

```text
Is relation internalization a possible path different from LLM-style text
prediction?
```

The honest answer was that V4-V6 were valuable but had shifted toward review
and governance shells. They tested whether relation evidence can be reviewed,
logged, replayed, and routed, but they did not build a new agent architecture.

Therefore the mainline was corrected:

```text
R1: Active Relation Internalization Agent
```

R1 changes the unit of evidence from review output back to agent behavior:

```text
active intervention
transition observation
relation-link learning
relation-guided action
counterfactual simulation
internal relation editing
OOD shortcut rejection
```

This does not prove a new intelligence paradigm. It creates the first runnable
foothold for testing whether a non-LLM agent can carry usable internal relations
as its operating substrate.

Current R1 result:

```text
random: gated_r1_score = 0.000
shortcut: gated_r1_score = 0.000
passive_memory: gated_r1_score = 0.000
relation_agent: gated_r1_score = 0.972
```

R1.1 then hardened this against:

```text
nuisance/confounder candidates
process-rule reversal
intervention cost tradeoffs
expanded candidate relation links
no-exploration ablation
```

Current R1.1 result:

```text
random: hardening_r11_gated_score = 0.000
shortcut: hardening_r11_gated_score = 0.000
passive_memory: hardening_r11_gated_score = 0.000
relation_no_explore: hardening_r11_gated_score = 0.000
relation_agent: hardening_r11_gated_score = 1.000
```

The next work should weaken R1's remaining scaffolding: finite candidate links,
explicit nuisance labels, and scripted active exploration.

R1.2 then reduced those supports:

```text
TRUE_LINK candidates were replaced by observed-transition candidate enumeration
unmarked nuisance features were tested without passing nuisance labels to the agent
a synthetic new process link was introduced outside the original TRUE_LINKS set
active exploration became low-coverage adaptive exploration rather than a fixed schedule
```

Current R1.2 result:

```text
random: discovery_r12_gated_score = 0.000
shortcut: discovery_r12_gated_score = 0.000
passive_memory: discovery_r12_gated_score = 0.000
relation_agent: discovery_r12_gated_score = 0.000
discovery_relation_agent: discovery_r12_gated_score = 1.000
```

R1.2 still does not reach unrestricted causal discovery. It keeps a small
process-variable schema and deterministic dynamics. That makes the next stage
clear: R2 should add partial observability, missing/noisy variables, and
relation-specific uncertainty inside the active agent.

R2 implemented that next step:

```text
missing observations
noisy and conflicting observations
inspect-before-action under relation uncertainty
unsafe automation rate
relation-specific uncertainty audit
```

Current R2 result:

```text
random: partial_r2_gated_score = 0.000
shortcut: partial_r2_gated_score = 0.000
passive_memory: partial_r2_gated_score = 0.000
discovery_relation_agent: partial_r2_gated_score = 0.000
uncertainty_discovery_agent: partial_r2_gated_score = 0.982
```

The key R2 distinction is:

```text
relation discovery is not enough under partial observability; the agent also
needs uncertainty-aware inspection and relation-specific audit.
```

## 14. Working Rule

The discussion source matters, but it is not evidence.

The rule for future work is:

```text
Discussion creates hypotheses.
Code creates tests.
Metrics create evidence.
Reports define claims.
Failure cases define boundaries.
```
