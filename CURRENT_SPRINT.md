# Current Sprint

Date: 2026-04-29

## Sprint R1: Relation Internalization Agent

## Goal

Return from review/governance shells to the core research line:

```text
Can a non-LLM agent actively learn usable internal relations through interaction?
```

R1 is an agent-body experiment, not a review text scorer and not a governance
wrapper.

## Why This Sprint Exists

V1 through V3 established relation diagnostics:

```text
prediction is not relation internalization
temporal prediction is not temporal relation internalization
uncertainty-aware takeover is required for noisy partial observations
```

V4 through V6 were useful shell stages:

```text
bounded review case
governance log and replay shell
multi-party audit resolution
```

But they moved away from the original ambition:

```text
relation internalization as a possible non-LLM intelligence route
```

R1 corrects that drift.

## Sprint Claim

```text
In a toy process world, a non-LLM relation agent can actively explore, recover
relation links, use them for action, answer counterfactuals, and change behavior
after internal relation edits.
```

## Sprint Boundary

Supported:

```text
toy active relation learning
relation-guided action
counterfactual relation simulation
internal relation editing
OOD warning-shortcut rejection
negative-control separation
```

Unsupported:

```text
LLM replacement
general intelligence
real engineering competence
real slope safety prediction
unrestricted world modeling
deployment readiness
```

## Tasks

### Done

```text
relation-agent-r1 created
process world implemented
active relation agent implemented
random baseline implemented
shortcut baseline implemented
passive-memory baseline implemented
relation recovery metric implemented
counterfactual metric implemented
edit-success metric implemented
active-exploration metric implemented
OOD warning shortcut test implemented
R1 report, claims, limitations, and self-audit generated
R1.1 nuisance/confounder hardening implemented
R1.1 process-rule reversal adaptation implemented
R1.1 intervention cost tradeoff implemented
R1.1 candidate expansion precision implemented
R1.1 no-exploration ablation implemented
R1_1_HARDENING_REPORT.md generated
R1_1_SELF_AUDIT.md generated
R1.2 discovery agent implemented
R1.2 unmarked nuisance rejection implemented
R1.2 synthetic new-link discovery implemented
R1.2 adaptive low-coverage exploration implemented
R1_2_DISCOVERY_REPORT.md generated
R1_2_SELF_AUDIT.md generated
R2 partial observability implemented
R2 missing observation tests implemented
R2 noisy/conflicting observation tests implemented
R2 inspect-before-action policy implemented
R2 unsafe automation metric implemented
R2 relation-specific uncertainty audit implemented
R2_PARTIAL_OBSERVABILITY_REPORT.md generated
R2_SELF_AUDIT.md generated
LLM black-box diagnostic sidecar implemented and frozen as negative baseline
PLOS-Test B-line diagnostic implemented
```

### Now

```text
Treat R1 as the current core agent line.
Keep V4-V6 as shell/evaluation lines, not the main intelligence route.
Treat R1.1 as the current hardening baseline.
Treat R1.2 as the current discovery baseline.
Treat R2 as the current partial-observability baseline.
Treat llm-relation-diagnostic as a frozen sidecar negative baseline, not the
mainline.
Treat PLOS-Test as a parallel B-line diagnostic for W -> O1 -> O2, not as a
generic video prediction project and not as an LLM/language route.
```

### Next

```text
R2.1:
  harden uncertainty behavior against inspect-overuse
  add irrelevant missing variables
  add benign noise cases that should not trigger inspect
  add audit-label-only negative control
  add fake uncertainty shortcut
```

### Sidecar Freeze

```text
llm-relation-diagnostic is frozen.
```

It showed that tested local black-box LLMs can answer some simple relation
prompts but fail the full gates for budgeted inspect, behavior-level local edit,
and exact audit. This is useful as a negative baseline, but continuing to add
prompt stress cases is no longer the main route.

Read:

```text
llm-relation-diagnostic\reports\LLM_RELATION_DIAGNOSTIC_FREEZE_MEMO.md
```

### B-Line PLOS-Test

```text
prelinguistic-operational-structure-test is implemented.
pytest -q: 20 passed
substrate expansion: predictive_coding_model, patch_graph_model,
koopman_model, and flow_checkpoint_model added
flow_checkpoint_model qualifies under current v1 gates:
plos_candidate_score = 0.434
flow_checkpoint_hardening_score = 0.771
```

It tests whether a model can form pre-linguistic operational structure from
continuous 2D dynamics without language labels. The evidence rule is behavior +
structural intervention + OOD. The current substrate search found
flow_checkpoint_model as the first PLOS candidate under the v1 gates. This is
still discounted because it uses a high checkpoint-selection prior and weak
causal-drop intervention magnitudes. The first hardening pass survives
noncausal decoy patch, static decoy, and causal endpoint-shift checks.

## Verification

```text
pytest -q: 5 passed
random: gated_r1_score = 0.000
shortcut: gated_r1_score = 0.000
passive_memory: gated_r1_score = 0.000
relation_agent: gated_r1_score = 0.972
```

## R1.1 Verification

```text
pytest -q: 9 passed
random: hardening_r11_gated_score = 0.000
shortcut: hardening_r11_gated_score = 0.000
passive_memory: hardening_r11_gated_score = 0.000
relation_no_explore: hardening_r11_gated_score = 0.000
relation_agent: hardening_r11_gated_score = 1.000
```

## R1.2 Verification

```text
pytest -q: 13 passed
random: discovery_r12_gated_score = 0.000
shortcut: discovery_r12_gated_score = 0.000
passive_memory: discovery_r12_gated_score = 0.000
relation_agent: discovery_r12_gated_score = 0.000
discovery_relation_agent: discovery_r12_gated_score = 1.000
```

## R2 Verification

```text
pytest -q: 17 passed
random: partial_r2_gated_score = 0.000
shortcut: partial_r2_gated_score = 0.000
passive_memory: partial_r2_gated_score = 0.000
discovery_relation_agent: partial_r2_gated_score = 0.000
uncertainty_discovery_agent: partial_r2_gated_score = 0.982
```

## Current Judgment

```text
R1 is valuable because it returns the program to agent internals rather than
review prose or governance wrappers. It is still a small deterministic toy, so
the result is a first runnable foothold, not a proof of a new intelligence
paradigm. R1.1 strengthened the foothold. R1.2 removes the hand-written true-link
candidate table. R2 shows that relation discovery is still insufficient under
partial observability unless the agent can inspect, avoid unsafe automation, and
name the uncertain relation link. The LLM black-box sidecar is useful for
negative evidence, but it is not a constructive route toward relation
internalization. PLOS-Test opens a separate B-line for continuous pre-linguistic
operational structure. Its current positive result is a candidate foothold:
flow_checkpoint_model passes the v1 gates and the first hardening pass, but the
pass is not blank-slate emergence and still needs broader hardening against
checkpoint-prior, metric-design, and richer OOD false positives.
```

## Decision Rule

If future work starts optimizing review text, reports, or governance logs without
improving the agent's internal relation learning and action use, stop and return
to R1/R1.1.
