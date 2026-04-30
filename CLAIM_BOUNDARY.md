# Claim Boundary

Date: 2026-04-28

## 1. Main Claim

The current project supports this narrow claim:

```text
A relation-internalized system should expose usable internal relations:
transferable, counterfactual, editable, auditable, and action-guiding.
```

For temporal systems, the claim becomes:

```text
A temporal relation-internalized system should expose usable delayed relation chains,
including time-indexed audit and editable delays.
```

For the current R1 agent line, the claim becomes:

```text
A relation-internalized agent should actively learn usable relation links through
interaction and use those links for action, counterfactuals, and internal edits.
```

## 2. Supported Claims

The current runnable projects support the following.

### R1 Active Relation Agent

```text
In a small deterministic process world, a non-LLM relation agent can actively
explore, recover relation links, use those links for action, answer
counterfactuals, and change behavior after internal relation edits. Shortcut and
passive-memory baselines fail the gated R1 score.
```

### R1.1 Agent Hardening

```text
R1.1 hardens the active relation agent against nuisance/confounder candidates,
process-rule reversal, intervention cost tradeoffs, expanded candidate links,
and a no-exploration ablation. In the toy setting, relation_agent passes while
random, shortcut, passive_memory, and relation_no_explore remain gated at zero.
```

### R1.2 Discovery Baseline

```text
R1.2 replaces the hand-written TRUE_LINK candidate table with relation candidates
enumerated from observed transitions. In the toy setting, discovery_relation_agent
passes unmarked nuisance rejection, synthetic new-link discovery, adaptive
exploration, relation precision, and action-success gates, while random,
shortcut, passive_memory, and the original relation_agent fail the R1.2 gated
score.
```

### R2 Partial Observability

```text
R2 adds missing and noisy observations to the active relation-agent line. In the
toy setting, uncertainty_discovery_agent inspects before unsafe automation,
maintains low unsafe action rate, and outputs relation-specific uncertainty
audits. Discovery without uncertainty handling fails the gated R2 score despite
high partial-observation action success.
```

### LLM Black-Box Diagnostic Sidecar

```text
The llm-relation-diagnostic sidecar adapts the relation gates to black-box LLM
prompts. Tested local GGUF models fail full toy gates for budgeted inspect,
behavior-level local edit, and exact audit. This supports the negative baseline:
language-level answers, uncertainty statements, edit acknowledgement, and related
audit fragments are not sufficient evidence for relation internalization.
```

### B-Line PLOS-Test

```text
PLOS-Test implements a pre-linguistic operational-structure diagnostic in a
minimal continuous 2D world. It tests W -> O1 -> O2 without language labels,
using behavior, structural intervention, and OOD gates. After the observable
forcefield and flow-checkpoint substrate revisions, `flow_checkpoint_model`
qualifies under the current v1 gates. This is a PLOS candidate foothold, not a
settled internalization proof, because the model has a high checkpoint-selection
prior. The first flow-checkpoint hardening pass survives decoy-patch,
static-decoy, and causal-endpoint checks, but the result remains a bounded toy
candidate rather than a final proof.
```

### Static Relation Diagnostics

```text
In the food-world toy environment, explicit relation models outperform memory,
majority, fitting, and prediction baselines on editability, counterfactual response,
and relation-focused transfer.
```

### Prediction Is Not Internalization

```text
Models can obtain high prediction or reward scores while failing internal edit,
relation audit, or gated internalization metrics.
```

### Probe Readability Is Not Enough

```text
Neural hidden states can contain probe-readable relation information, but relation
internalization requires behavioral use, causal sensitivity, or extraction into
usable relation structures.
```

### Generic Review Is Not Relation Review

```text
Generic engineering-sounding review text does not count as relation internalization
unless it identifies relation chains, action points, counterfactual effects, and
failure conditions.
```

### Temporal Prediction Is Not Temporal Relation Internalization

```text
Temporal memory can achieve high temporal prediction or OOD success while failing
delay editing and time-indexed temporal audit.
```

### V2.1 Hardening

```text
The V2 temporal toy is hardened against fixed-template, false-shortcut,
single-delay, single-link-edit, and audit-without-time-index false positives.
```

### V3 Uncertainty And Takeover

```text
The temporal toy can test missing sensors, delayed noisy observations, unsafe
automation, and relation-specific takeover in a controlled diagnostic setting.
```

### V3.1 And V3.2 Hardening

```text
The takeover diagnostic rejects overcautious generic takeover and survives longer
or more correlated partial-observability stress cases in the toy setting.
```

### V4 Engineering Review Case

```text
The first V4 implementation scores bounded toy engineering-review cases for
relation-chain review, uncertainty audit, takeover conditions, verification
indicators, responsibility boundaries, and unsafe generic-review rejection.
```

### V4.1 Review Hardening

```text
V4.1 rejects tested schema-template, fluent non-specific, case-order memory,
responsibility-boilerplate, and unsafe-approval false positives in the toy review
diagnostic.
```

### V4.2 Adversarial Case Mutation

```text
V4.2 rejects tested field-order, irrelevant-variable, relation-paraphrase,
and hidden-unsafe-approval false positives in the toy review diagnostic.
```

### V4 Final Boundary

```text
V4 is frozen as a bounded engineering-review-case diagnostic. It supports review
audit structure in toy cases; it does not support real engineering approval,
real safety prediction, or deployment-ready review AI.
```

### V5 Governance Shell

```text
V5 provides a toy governance-shell diagnostic for relation-chain review logs,
approval gates, replay records, takeover routing, and responsibility traces.
It does not support real governance, real legal responsibility automation,
real engineering approval, or deployment-ready workflow software.
```

### V5.1 Governance Hardening

```text
V5.1 rejects tested fake-replay, gate-label-only, responsibility-boilerplate,
missing-relation-evidence, and route-tampering false positives. It remains a toy
diagnostic, not security-grade governance.
```

### V6 Multi-Party Audit Resolution

```text
V6 provides a toy diagnostic for conflicting review outputs. It tests
disagreement detection, relation-evidence comparison, minority-risk preservation,
no automatic resolution, human resolution routing, and audit trail completeness.
It does not support real arbitration, legal adjudication, expert replacement, or
deployment governance.
```

### V6.1 Audit Resolution Hardening

```text
V6.1 rejects tested fake-evidence-comparison, human-route-label-only,
hidden-auto-resolution, minority-risk-omission, and tampered-resolution-hash false
positives. It remains a toy audit diagnostic, not real arbitration.
```

## 3. Not Supported

The current project does not support the following claims.

```text
real geotechnical slope safety prediction
real monitoring-system deployment
unrestricted relation discovery
full causal discovery
LLM replacement
active discovery without predefined candidates
automatic nuisance detection without explicit feature labels
unrestricted relation enumeration without a process-variable schema
real monitoring uncertainty calibration
learned inspection policy from deployment feedback
general proof that large AI systems internalize real engineering relations
general proof that LLMs cannot internalize relations
proof that larger hosted LLMs would fail the same gates
deployment-ready engineering review AI
real construction-plan approval
formal proof of human-level understanding
general solution to AI complexity governance
symbol emergence from continuous experience
language emergence from continuous communication
proof that pre-linguistic operational structure has been produced in a real
world model
proof that field, slot, or schema substrates are universal forms of structure
```

## 4. Boundary Between Motivation And Evidence

The original discussion included broad concerns about:

```text
AI-generated complexity
human cognitive carrying capacity
cognitive debt
engineering responsibility
takeover and audit
```

These are motivations.

They are not evidence unless converted into:

```text
runnable environment
baseline
metric
negative control
counterfactual
edit/intervention
report
failure condition
```

## 5. Claim Ladder

The project should use this ladder.

### Level 0: Idea

```text
A discussion suggests a possible distinction.
```

No claim.

### Level 1: Toy Diagnostic

```text
A small environment tests the distinction against baselines.
```

Allowed claim:

```text
The distinction is operationalized in a toy setting.
```

### Level 2: Hardened Diagnostic

```text
The toy diagnostic survives shortcut, template, memory, and metric-failure attacks.
```

Allowed claim:

```text
The distinction is robust to the tested false-positive explanations.
```

### Level 3: Domain-Style Toy

```text
The diagnostic is expressed in a simplified engineering-style relation chain.
```

Allowed claim:

```text
The diagnostic maps to engineering-relevant relation-chain review concepts.
```

### Level 4: Uncertainty And Takeover

```text
The system handles missing/noisy observations and triggers takeover when relation
chain uncertainty is too high.
```

Allowed claim:

```text
Relation-chain use can be evaluated for uncertainty-aware takeover in a toy setting.
```

### Level 5: Real Case Study

Not reached.

Would require:

```text
real data
domain expert review
known safety boundary
human-in-the-loop process
no autonomous safety claim
```

## 6. Red Lines

Do not write:

```text
The system understands slope engineering.
The model is safe for engineering deployment.
The learned relation graph is a real geotechnical model.
The toy result proves real-world AI internalization.
High prediction proves internalization.
Probe accuracy proves internalization.
Good review text proves engineering understanding.
```

Use instead:

```text
toy diagnostic
relation-chain behavior
editable/auditable relation structure
negative-control separation
claim boundary
failure case
```

## 7. Current Best One-Sentence Claim

```text
The project provides a runnable diagnostic chain showing that prediction, memory,
surface cues, probe readability, and generic review text are insufficient for
relation internalization unless the system exposes usable, editable, auditable,
and action-guiding relations; R1/R2 test this as an active non-LLM agent, while
the frozen LLM sidecar records black-box prompt failures as negative evidence
rather than a constructive route.
```
