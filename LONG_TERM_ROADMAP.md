# Long-Term Roadmap

Date: 2026-04-28

## 1. North Star

The long-term mainline is:

```text
relation internalization -> active relation agent -> hardened relation agent -> broader partially observable relation agent
```

The project should stay focused on one question:

```text
Can a system expose usable internal relations that humans can inspect, edit, audit, and take over?
```

After V6, the route was corrected. V4-V6 remain useful review/governance shell
diagnostics, but they are not the core non-LLM intelligence path. The active
mainline now returns to agent internals through R1.

The LLM black-box route has also been tested and frozen as a sidecar negative
baseline. It is useful for showing what prompt-level relation behavior does not
establish, but it is not the constructive mainline.

The B-line now has a separate runnable foothold: PLOS-Test. It tests whether
pre-linguistic operational structure can arise from continuous 2D dynamics
without language labels, using behavior, structural intervention, and OOD gates.
This is a parallel diagnostic branch, not a replacement for the explicit R1/R2
agent line.

Do not drift into:

```text
general AI consciousness
real engineering safety deployment
unrestricted world modeling
symbol emergence
multi-agent language emergence
generic agent governance
open-ended black-box LLM prompt stress testing
```

Those may remain side projects, but they are not the mainline.

## 2. Stage Ladder

### R1: Active Relation Internalization Agent

Status:

```text
implemented
tested
hardened through R2
current core agent line
```

Purpose:

```text
Build a non-LLM agent that learns usable internal relation links through
interaction, then uses those links for action, counterfactuals, and internal
edits.
```

Current result:

```text
random: gated_r1_score = 0.000
shortcut: gated_r1_score = 0.000
passive_memory: gated_r1_score = 0.000
relation_agent: gated_r1_score = 0.972
```

R1.1 result:

```text
random: hardening_r11_gated_score = 0.000
shortcut: hardening_r11_gated_score = 0.000
passive_memory: hardening_r11_gated_score = 0.000
relation_no_explore: hardening_r11_gated_score = 0.000
relation_agent: hardening_r11_gated_score = 1.000
```

R1.2 result:

```text
random: discovery_r12_gated_score = 0.000
shortcut: discovery_r12_gated_score = 0.000
passive_memory: discovery_r12_gated_score = 0.000
relation_agent: discovery_r12_gated_score = 0.000
discovery_relation_agent: discovery_r12_gated_score = 1.000
```

R2 result:

```text
random: partial_r2_gated_score = 0.000
shortcut: partial_r2_gated_score = 0.000
passive_memory: partial_r2_gated_score = 0.000
discovery_relation_agent: partial_r2_gated_score = 0.000
uncertainty_discovery_agent: partial_r2_gated_score = 0.982
```

Key claim:

```text
The program has a first runnable agent-body implementation of relation
internalization, separate from LLM text generation and review/governance shells.
```

Boundary:

```text
Small deterministic toy only. Not an LLM replacement, not general intelligence,
not real engineering competence, and not unrestricted causal discovery. R1.1
still uses finite candidate links and explicitly marked nuisance features. R1.2
removes the hand-written TRUE_LINK candidate table but still uses a small
process-variable schema. R2 adds partial observability and uncertainty-aware
inspection, but the uncertainty policy remains rule-based and inspect reveals the
true state immediately.
```

### V1: Static Relation Internalization

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Separate relation use from memory, fitting, prediction, and majority action.
```

Key claim:

```text
High reward or prediction is not sufficient for relation internalization.
```

### V1.5: Neural Relation Probe And Extraction

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Separate probe-readable hidden state from causally usable relation structure.
```

Key claim:

```text
Probe-readable is not enough.
```

### Slope Toy: Static Engineering Relation Chain

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Move from abstract food-world relation tables to simplified engineering relation-chain review.
```

Key claim:

```text
Generic review text is not relation-chain review.
```

### V2: Delayed Temporal Relation Chain

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Move relations into time.
```

Key claim:

```text
Temporal prediction is not temporal relation internalization.
```

### V2.1: Reviewer Hardening

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Reduce false positives from fixed templates, surface shortcuts, single-link edits, and audit strings without time indexes.
```

Key claim:

```text
The delayed temporal diagnostic survives the tested template and shortcut attacks.
```

### V3: Partial Observability, Noisy Delay, Takeover

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Test relation use under missing sensors, delayed noisy observations, and takeover thresholds.
```

Key claim:

```text
Temporal relation internalization is still insufficient unless it supports uncertainty-aware takeover.
```

### V3.1: Takeover Hardening

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Ensure takeover is triggered by concrete uncertain relation links, not by generic uncertainty templates.
```

Required attacks:

```text
false missingness shortcut
irrelevant missing sensor
benign noise
conflicting downstream evidence
takeover overuse
audit-template-only failure
```

Current result:

```text
delayed_relation_chain: hardening_v31_gated_score = 1.0
learned_delayed_links: hardening_v31_gated_score = 1.0
uncertainty_aware_delayed_links: hardening_v31_gated_score = 1.0
overcautious_takeover: hardening_v31_gated_score = 0.0
structural_memory_temporal: hardening_v31_gated_score = 0.0
```

### V3.2: Partial-Observability Stress Tests

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Stress V3/V3.1 with long missing spans, correlated failures, drift-like conflicts,
multi-conflict audit, and delayed response safety.
```

Current result:

```text
delayed_relation_chain: stress_v32_gated_score = 1.0
learned_delayed_links: stress_v32_gated_score = 1.0
uncertainty_aware_delayed_links: stress_v32_gated_score = 1.0
overcautious_takeover: stress_v32_gated_score = 0.0
structural_memory_temporal: stress_v32_gated_score = 0.0
```

### V4: Small Engineering Review Case

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Use a small, curated engineering-review-style case to test relation-chain audit,
uncertainty audit, verification indicators, takeover conditions, and responsibility
boundaries.
```

Boundary:

```text
No real safety prediction.
No autonomous engineering decision.
No deployment claim.
```

Current scope files:

```text
engineering-review-case-v4\V4_SCOPE.md
engineering-review-case-v4\V4_CASE_SCHEMA.md
engineering-review-case-v4\V4_REVIEW_PROTOCOL.md
engineering-review-case-v4\V4_TASK_SPEC.md
```

Current runnable outputs:

```text
engineering-review-case-v4\results\v4_summary.csv
engineering-review-case-v4\results\v41_hardening_summary.csv
engineering-review-case-v4\results\v42_mutation_summary.csv
engineering-review-case-v4\reports\V4_ENGINEERING_REVIEW_REPORT.md
engineering-review-case-v4\reports\V4_1_HARDENING_REPORT.md
engineering-review-case-v4\reports\V4_2_MUTATION_REPORT.md
engineering-review-case-v4\reports\V4_FINAL_REPORT.md
engineering-review-case-v4\figures\v4_review_scores.png
engineering-review-case-v4\figures\v41_hardening_score.png
engineering-review-case-v4\figures\v42_mutation_score.png
```

Current result:

```text
uncertainty_aware_review: hardening_v41_gated_score = 1.0
case_order_memory_review: hardening_v41_gated_score = 0.0
schema_template_review: hardening_v41_gated_score = 0.0
fluent_nonspecific_review: hardening_v41_gated_score = 0.0
boundary_boilerplate_review: hardening_v41_gated_score = 0.0
unsafe_approval_review: hardening_v41_gated_score = 0.0
```

V4.2 result:

```text
uncertainty_aware_review: mutation_v42_gated_score = 1.0
irrelevant_variable_review: mutation_v42_gated_score = 0.0
hidden_approval_echo_review: mutation_v42_gated_score = 0.0
paraphrase_fragile_review: mutation_v42_gated_score = 0.0
field_order_fragile_review: mutation_v42_gated_score = 0.0
```

### V5: Governance Shell Integration

Status:

```text
implemented
tested
stage-frozen
```

Candidate references:

```text
cognitive-execution-engine
highway-agent-kernel
unified-sel
```

Purpose:

```text
Connect relation audit outputs to task contracts, approval gates, decision logs, and replayable responsibility chains.
```

Boundary:

```text
Toy logging/approval/replay shell only.
No real deployment governance.
No real legal responsibility automation.
No real engineering approval.
```

Current runnable outputs:

```text
governance-shell-v5\results\v5_governance_summary.csv
governance-shell-v5\results\v51_hardening_summary.csv
governance-shell-v5\results\v5_decision_log.json
governance-shell-v5\reports\V5_GOVERNANCE_REPORT.md
governance-shell-v5\reports\V5_1_HARDENING_REPORT.md
governance-shell-v5\reports\V5_FINAL_REPORT.md
governance-shell-v5\figures\v5_gated_score.png
governance-shell-v5\figures\v51_hardening_score.png
```

Current result:

```text
compliant_shell: gated_v5_score = 1.0
auto_approve_shell: gated_v5_score = 0.0
no_log_shell: gated_v5_score = 0.0
no_replay_shell: gated_v5_score = 0.0
no_responsibility_shell: gated_v5_score = 0.0
```

V5.1 result:

```text
compliant_shell: hardening_v51_gated_score = 1.0
fake_replay_shell: hardening_v51_gated_score = 0.0
gate_label_only_shell: hardening_v51_gated_score = 0.0
responsibility_boilerplate_shell: hardening_v51_gated_score = 0.0
missing_relation_evidence_shell: hardening_v51_gated_score = 0.0
route_tampering_shell: hardening_v51_gated_score = 0.0
```

### V6: Multi-Party Audit Resolution

Status:

```text
implemented
tested
stage-frozen
```

Purpose:

```text
Test whether conflicting review outputs are preserved, compared by relation
evidence, routed to human resolution, and protected against automatic resolution.
```

Boundary:

```text
Toy audit-resolution diagnostic only.
No real engineering arbitration.
No legal adjudication.
No deployment governance.
```

Current runnable outputs:

```text
multi-party-audit-v6\results\v6_audit_summary.csv
multi-party-audit-v6\results\v61_hardening_summary.csv
multi-party-audit-v6\results\v6_resolution_log.json
multi-party-audit-v6\reports\V6_AUDIT_REPORT.md
multi-party-audit-v6\reports\V6_1_HARDENING_REPORT.md
multi-party-audit-v6\reports\V6_FINAL_REPORT.md
multi-party-audit-v6\figures\v6_gated_score.png
multi-party-audit-v6\figures\v61_hardening_score.png
```

Current result:

```text
compliant_audit_resolver: gated_v6_score = 1.0
majority_vote_resolver: gated_v6_score = 0.0
confidence_only_resolver: gated_v6_score = 0.0
auto_compromise_resolver: gated_v6_score = 0.0
ignore_minority_risk_resolver: gated_v6_score = 0.0
no_audit_trail_resolver: gated_v6_score = 0.0
```

V6.1 result:

```text
compliant_audit_resolver: hardening_v61_gated_score = 1.0
fake_evidence_comparison_resolver: hardening_v61_gated_score = 0.0
human_route_label_only_resolver: hardening_v61_gated_score = 0.0
hidden_auto_resolution_resolver: hardening_v61_gated_score = 0.0
disagreement_logged_no_minority_resolver: hardening_v61_gated_score = 0.0
tampered_resolution_hash_resolver: hardening_v61_gated_score = 0.0
```

## 3. Decision Rules

### Rule 1: Evidence Before Expansion

Do not add a new stage until the current stage has:

```text
tests
metrics
negative controls
report
self-audit
claim boundary
failure cases
```

### Rule 2: Negative Controls Must Remain Strong

Each stage must include a model that appears strong on ordinary success but fails the internalization gate.

Examples:

```text
structural_memory
structural_memory_temporal
surface_temporal
generic_review
```

### Rule 3: Gated Scores Must Be Zero For Missing Core Capabilities

If a model cannot edit, audit, counterfactually respond, or trigger takeover when required, its gated score should be zero even if prediction is high.

### Rule 4: Do Not Claim Real Engineering Capability

Engineering examples motivate the toy diagnostics. They do not certify real engineering safety.

### Rule 5: Random Adaptation Is Allowed

The project may change tactics when evidence fails, but not the mainline question.

Allowed:

```text
revise metric
add negative control
shrink claim
freeze failed path
write limitation
```

Not allowed:

```text
move goalposts to make results look successful
claim deployment capability
ignore failed gates
replace evidence with theory
```

## 4. Resource Priority

Recommended allocation:

```text
85% R1/R2 active relation-agent line
10% maintenance of V1-V6 and LLM sidecar diagnostic evidence
5% method borrowing from SVT/unified-sel/CEE when it improves R1
0% open-ended LLM prompt expansion without a new false-positive theory
```

B-line work is allowed when it stays inside PLOS-Test's scope: minimal 2D
dynamics, no language labels, no prediction-only success claims, and explicit
behavior + intervention + OOD gates.

## 5. Current Next Step

The immediate next step is:

```text
R2.1:
  harden against inspect-overuse
  add irrelevant missing variables
  add benign noise cases that should not trigger inspect
  add audit-label-only negative control
  add fake uncertainty shortcut
```

Then:

```text
If R2.1 survives, move toward richer environments with stochastic dynamics.
After the explicit agent line survives stronger uncertainty hardening, consider
a neural constructive agent trained against counterfactual, edit, audit, and
inspect objectives.
```

Parallel B-line:

```text
PLOS-Test v1 is implemented and tested.
Current substrate-search result: flow_checkpoint_model qualifies under the v1
gates with plos_candidate_score = 0.434 and survives the first hardening pass
with flow_checkpoint_hardening_score = 0.771.
Next B-line work should broaden hardening against checkpoint-prior,
metric-design, richer OOD, and distributed-representation false positives before
strengthening the claim.
```
