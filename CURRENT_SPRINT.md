# Current Sprint

Date: 2026-05-04

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
relation-specific uncertainty and inspection
cost-aware active inspection
```

Unsupported:

```text
LLM replacement
general intelligence
real engineering competence
real slope safety prediction
unrestricted world modeling
deployment readiness
learned inspection policy (still rule-based)
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
R2.1 uncertainty hardening implemented
R2.1 inspect-overuse penalty implemented
R2.1 irrelevant missing variable rejection implemented
R2.1 benign noise rejection implemented
R2.1 audit-label-only negative control implemented
R2.1 fake uncertainty shortcut rejection implemented
R2.1 inspection cost and budget constraints implemented
R2.1 non-oracle inspection (single field reveal) implemented
R2_1_HARDENING_REPORT.md generated
R2_1_SELF_AUDIT.md generated
R3 active inspection implemented
R3 multi-field target selection implemented
R3 information gain efficiency implemented
R3 budgeted safe action implemented
R3 overinspection penalty implemented
R3 sequential update accuracy implemented
R3 report, self-audit generated
LLM black-box diagnostic sidecar implemented and frozen as negative baseline
PLOS-Test B-line diagnostic implemented through B6.4
G1 minimal generator implemented and run
G1.1 pressure hardening implemented and run
G1.2 feature induction implemented and run
G1 adversarial review generated
G1.2 adversarial review generated
G1.2-Clean oracle-free selection implemented and run
G1.3 feature discovery implemented and run
G2 compositional world implemented and run
G2.1 indirect pathway hardening implemented and run
```

### Now

```text
Treat R1 as the current core agent line.
Treat R2.1 (relation-specific uncertainty) as the current uncertainty hardening baseline.
Treat R3 (active inspection) as the current inspection baseline.
Keep V4-V6 as shell/evaluation lines, not the main intelligence route.
Treat R1.1 as the hardening baseline.
Treat R1.2 as the discovery baseline.
Treat R2 as the partial-observability baseline.
Treat llm-relation-diagnostic as a frozen sidecar negative baseline, not the mainline.
Treat PLOS-Test as a parallel B-line diagnostic for W -> O1 -> O2.
B-line is currently at B6.4 stage (risk-constrained closed-loop with combined remap hardening).
B6 cannot yet gate to B7 due to combined remap limitation (combined_policy_score = 0.675).
```

### Next

R line:

```text
R4: Learned Inspection Policy
  replace rule-based inspection values with learned uncertainty/value estimation
  preserve existing gates (precision, cost, budget, unsafe automation)
  test whether learned policy outperforms hand-shaped relation_specific_uncertainty_agent
```

B line:

```text
B6.4.3: Dynamics-Delay-Credit Interaction Refinement
  isolate the combined remap failure source
  reduce combined oracle gap (currently 0.325)
  either reduce or explicitly bound candidate-search dependence under combined remap
  goal: either fix the gap or accept it as a known boundary before B7
```

Cross-cutting:

```text
Paper: integrated B-line PLOS-Test results through B6.4 — DONE
Paper: integrated G-line results G1 through G2.1 — DONE
G-line complete results:
  G1: 0.890 (OOD 0.868), rule avoids feedback/compression → shortcut
  G1.1: 0.597 (OOD 0.779), degeneracy audit: ablations can outperform full generator
  G1.2: 0.813 (OOD 0.887), oracle leakage in program selection (forbidden evaluator_ground_truth)
  G1.2-Clean: 0.740 (OOD 0.835), oracle-free, feedback drop 0.289 preserved
  G1.3: 0.276 (OOD 0.267), passive k-means feature discovery fails
  G2: 0.457 (OOD 0.335, ARI 0.610), indirect pathway unnecessary (drop=0.000)
  G2.1: 0.461 (OOD 0.335, ARI 0.610), forced indirect → kill drop 0.005, replicates G1.1 degeneracy

G-line verdict: discrete grid search over hand-decoded features cannot produce
  genuinely internalized operational structure. In 7 rounds across 2 structural
  channels (feedback/compression and cross-object indirect), every round produces
  operational-looking output that fails degeneracy audit: the target channel is
  constructed but never causally used. The pattern is systematic, not accidental.

R-line verdict: active interaction + schema-given relation space → 1.000.
  Structure can be discovered through interaction when the dimensional schema
  is provided; it cannot be generated from scratch in the current architecture.

Next direction options:
  A) Write/finalize paper with current evidence (complete as-is)
  B) G2.2: continuous PDE world (real physics, no hand-designed latent profiles)
  C) Neural constructive agent (replace discrete grid search with gradient-based learning)
  **DONE — C turned into PLOS-Test causal graph world + 35 experiments.**

PLOS-Test exploration complete — 35 experiments total across 3 paradigms:
  Physics world (21 runs): cross benefit ceiling ≈0.014, TE efficiency max 7%
  Causal graph world (8 runs): architecture-level emergence proven (benefit=0.041, +57%)
  Architecture variants (6 runs): content-level emergence proven IMPOSSIBLE under gradient training

**KEY FINDING: Architecture-level vs Content-level Emergence Boundary**
  Architecture-level: learning "message passing works" → ACHIEVED (OOD retains 116%)
  Content-level: learning "WHICH edges matter more" → IMPOSSIBLE (all weights → uniform 0.5)
  Root cause: O(1/N) gradient blindness — individual edge contribution ~1/N, 
    finite-difference gradient ≈10^-11 for float32 — invisible.
  Even per-edge architectures (EdgeSpecificGNN, HyperEdgeGNN, AttnRouterGNN) 
    all collapse to uniform routing.

**New exploration directions (2026-05-06):**
  A) Active intervention learner (push→observe→learn edges)
  B) LLM diagnostic on causal graph world (does GPT/Claude show same boundary?)
  C) Non-gradient discrete search (random delete→test loss→recover structure)
  D) Theory writeup: architecture vs content emergence as conceptual contribution
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
prelinguistic-operational-structure-test has advanced through B6.4.

B-line progression:
  B1: initial PLOS candidate search (flow_checkpoint_model qualifies)
  B2: delayed operational checkpoint substrate (trace-bearing models)
  B2.1: trace hardening against false/swap/deletion/noise/length attacks
  B2.1a: trace degeneracy audit (identical scores found)
  B2.2: trace selector disentanglement (shared-selector identified)
  B2.3: private trace selector construction (shared-selector resolved)
  B3: active inspection under budget (0.985)
  B3.1: inspection degeneracy audit (single dominant target found)
  B3.2: mechanism-disambiguated inspection (0.963)
  B4: intervention/action selection (0.985)
  B4.1: intervention degeneracy audit (fixed_action_type = 1.000)
  B4.2: action-type disambiguation (0.937, resolves B4.1 shortcut)
  B5: epistemic-pragmatic closed-loop operation (observe→inspect→update→intervene→feedback)
  B6: risk-constrained closed-loop (B6.1-B6.4 hardening series)

Current B6 status:
  B6.4 combined remap shows combined_policy_score = 0.675, oracle_gap = 0.325.
  B6_TO_B7_GATE_MEMO.md recommendation: STAY_IN_B6_REFINEMENT.
  Next: B6.4.3 Dynamics-Delay-Credit Interaction Refinement.
  B7 will require compositional scaling (multi-object/risk/delay/indirect-path toy worlds).
```

It tests whether a model can form pre-linguistic operational structure from
continuous 2D dynamics without language labels. The evidence rule is behavior +
structural intervention + OOD. The project theory freeze V2 identifies the
current bottleneck as the W → O₁ → O₂ closed-loop update challenge, which
maps to B6's combined remap limitation.

## R Line Verification Summary

R1:

```text
random: gated_r1_score = 0.000
shortcut: gated_r1_score = 0.000
passive_memory: gated_r1_score = 0.000
relation_agent: gated_r1_score = 0.972
```

R1.1:

```text
random: hardening_r11_gated_score = 0.000
shortcut: hardening_r11_gated_score = 0.000
passive_memory: hardening_r11_gated_score = 0.000
relation_no_explore: hardening_r11_gated_score = 0.000
relation_agent: hardening_r11_gated_score = 1.000
```

R1.2:

```text
random: discovery_r12_gated_score = 0.000
shortcut: discovery_r12_gated_score = 0.000
passive_memory: discovery_r12_gated_score = 0.000
relation_agent: discovery_r12_gated_score = 0.000
discovery_relation_agent: discovery_r12_gated_score = 1.000
```

R2:

```text
random: partial_r2_gated_score = 0.000
shortcut: partial_r2_gated_score = 0.000
passive_memory: partial_r2_gated_score = 0.000
discovery_relation_agent: partial_r2_gated_score = 0.000
uncertainty_discovery_agent: partial_r2_gated_score = 0.982
```

R2.1:

```text
missing_always_inspect: r2_1_gated_score = 0.000
relation_specific_uncertainty_agent: r2_1_gated_score = 0.933
```

R3:

```text
random_inspect_field: r3_gated_score = 0.000
first_missing_inspect: r3_gated_score = 0.000
missing_always_inspect: r3_gated_score = 0.000
risk_first_inspect: r3_gated_score = 0.000
active_inspection_agent: r3_gated_score = 0.933
```

G-line:

```text
G1: g1_generator_mean_score = 0.890 (OOD 0.868, oracle_gap 0.110, mask_f1 0.934)
G1.1: g1_1_mean_score = 0.597 (OOD 0.779, oracle_gap 0.302, mask_f1 0.533)
    degeneracy: no_feedback > generator on compression_required; no_compression > generator on feedback_required
G1.2: g1_2_mean_score = 0.813 (OOD 0.887, oracle_gap 0.085, mask_f1 0.750)
    feature necessity: feedback drop -0.530, compression drop -0.555 (condition-specific)
    caveat: training-time oracle leakage via evaluator_ground_truth in program selection
G1.2-Clean: g1_2_clean_mean_score = 0.740 (OOD 0.835, oracle_gap 0.159, mask_f1 0.635)
    oracle-free training objective using observed_reward only; feedback drop 0.289 preserved
G1.3: g1_3_mean_score = 0.276 (OOD 0.267, oracle_gap 0.622, mask_f1 0.079)
    FAILED — k-means clustering on raw vectors cannot recover actionable structure
    lesson: passive feature discovery insufficient; active/interaction-based construction needed
G2: g2_mean_score = 0.457 (OOD 0.335, ood_novel 0.506, ood_topology 0.606)
    ARI 0.610 (meets 0.60 target), edge_discovery_f1 1.000, oracle_gap 0.193
    degeneracy: indirect=False rule scores identically to indirect=True; cross-object edges unused
    surprise: OOD novel/topology scores BETTER than OOD remap — structure generalizes upward
```

## Current Judgment

```text
R1 through R3 form a complete diagnostic ladder from active relation learning
through partial observability, uncertainty hardening, and cost-aware active
inspection. All baselines remain gated at zero while positive agents score 0.933-1.000.

The R line has reached a natural checkpoint. All inspection and uncertainty
policies remain rule-based. The next scientific step is R4: replacing rule-based
inspection with learned uncertainty/value estimation while preserving all gates.

The B line PLOS-Test has advanced further than initially expected, reaching
B6.4 risk-constrained closed-loop with combined remap hardening. The combined
remap limitation (0.675) is the current bottleneck preventing B7 compositional
scaling.

The paper draft (MAIN_PAPER_DRAFT_V0.md) is substantially complete with all R-line
and V-line results. The main gap is integration of B-line PLOS-Test results as a
parallel diagnostic branch.
```

## Decision Rule

If future work starts optimizing review text, reports, or governance logs without
improving the agent's internal relation learning and action use, stop and return
to R1/R1.1.
