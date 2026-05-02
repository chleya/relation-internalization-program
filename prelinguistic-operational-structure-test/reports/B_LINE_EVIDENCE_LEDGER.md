# B-Line Evidence Ledger

## 1. Project Purpose

This project is not a standard toy control benchmark. It studies Pre-Linguistic Operational Structure: whether a system can form an internal operational structure that constrains action without language, symbolic labels, or human semantic shortcuts.

The operational structure under test is expected to constrain:

- inspect
- intervene
- abstain
- wait
- revise
- fallback risk inference
- delayed credit assignment
- structural repair

Core proposition:

Structure is not what predicts the clean future. Structure is what continues to constrain action under missing information, wrong cues, delayed consequences, risk conflict, and feedback pressure.

## 2. Evidence Ladder

### B5 - Closed-Loop Epistemic-Pragmatic Operation

B5 moved the project from one-shot action into a minimal closed loop:

```text
observe -> inspect -> update trace -> intervene -> observe consequence -> revise trace
```

Main contribution:

- separated epistemic value from pragmatic value
- added trace update after inspection
- added feedback revision after consequence observation
- established a toy closed-loop path rather than a one-step prediction or action diagnostic

Claim boundary:

- still a 64x64 toy world
- does not prove real active inference
- does not prove robot control
- does not prove human-like or language-free cognition

### B6 - Clean Risk-Constrained Actionability

B6 introduced explicit actionability constraints:

- observable / inspectable regions
- direct intervention
- indirect intervention
- abstain
- risk, cost, unsafe, and irreversible constraints

Main contribution:

- moved from action selection into risk-constrained actionability
- tested inspect / direct intervention / indirect intervention / abstain under an explicit operational mask

Key limitations:

- the public actionability mask exposed strong operational cues
- `previous_trace_state["region"]` remained a strong target prior
- `n_ood` existed as a config field but was not an active OOD split
- clean B6 did not prove robust risk intelligence

### B6.1 - Reviewer-Hardening Stress Diagnostics

B6.1 stress-tested B6 clean results with:

- noisy mask
- missing mask
- hidden irreversibility
- delayed indirect intervention
- inspect cost
- risk-reward conflict
- spurious safe cue
- leakage and metric-bug audit

Key results:

- hardening_policy_mean_score = 0.955
- hardening_policy_min_score = 0.725
- oracle_mean = 1.000
- risk_blind_mean = 0.617
- mask_only_mean = 0.780
- random_mean = 0.724
- always_abstain_mean = 0.622

B6.1 exposed:

- `missing_mask=1.0` reduced utility and increased abstention
- `delay_steps=5` had delayed_indirect_success_rate = 0
- clean masks and clean risk labels were still too strong as operational cues

### B6.2 - Fallback Risk and Delayed Credit Diagnostics

B6.2 first pass added:

- fallback risk inference
- delayed credit assignment
- wrong / missing / ambiguous / low-confidence trace stress
- hard_hidden_mask
- risk_reward_conflict

B6.2 second pass repaired the first-pass coverage gaps:

- missing_trace
- ambiguous_trace
- low_confidence_trace
- hard_hidden_mask
- risk_reward_conflict
- wrong_trace_state_ambiguous

Key second-pass results:

- delay5_true_success_score = 0.667
- risk_reward_conflict:
  - b62_policy = 1.000
  - risk_blind = 0.000
  - always_abstain = 0.450

Key limitations:

- wrong_trace remained unresolved:
  - b62_policy = 0.753
  - mask_only = 0.833
  - state_only = 1.000
- hide_indirect_target remained candidate-search fallback:
  - candidate_indirect_search_success_rate = 0.583
- private trace necessity was not proven

### B6.3 - Structural Necessity and Trace Repair Diagnostics

B6.3 introduced causal ablations to test whether B6.2 behavior depends on trace, history, feedback, delayed credit, and candidate search.

Key results:

- best_b6_3_structural_necessity_score = 0.997
- mean_b6_3_structural_necessity_score = 0.620
- trace necessity drop = 0.124
- candidate search drop = 0.127

Ablations causing drop:

- remove_risk_cue = 0.179
- remove_trace = 0.124
- shuffle_trace = 0.124
- corrupt_trace = 0.124
- disable_candidate_search = 0.127
- disable_inspection_recovery = 0.095

Weak or no-drop ablations:

- disable_delayed_credit_buffer = 0.006
- freeze_feedback_update = 0.019
- remove_history = 0.025
- hide_public_state_cue = 0.000

Conclusion:

- trace appeared partially necessary
- candidate search appeared necessary
- history necessity was not proven
- feedback update necessity was not proven
- delayed credit buffer necessity was weak in aggregate
- wrong_trace remained unresolved:
  - b63_policy = 0.753
  - state_only = 1.000
  - mask_only = 0.959

### B6.3.1 - Wrong-Trace and Mechanism Necessity Refinement

B6.3.1 targeted B6.3 blockers without entering B7.

Key results:

- best_b6_3_1_refinement_score = 1.000
- mean_b6_3_1_refinement_score = 0.987
- wrong_trace_no_public_state added
- state_only_drop_on_wrong_trace_no_public_state = 0.550
- feedback_required_trace_repair drop under freeze_feedback_update = 0.275
- history_required_* drop under remove_history = max 0.550
- delay5_credit_buffer_required drop under disable_credit_buffer = 0.550
- hidden_indirect_discovery_score = 1.000
- candidate_search_fallback_score = 0.000

Audit:

- forbidden_reference_count_max = 0
- policy_source_forbidden_reference_count = 0
- poisoned evaluator invariance passed
- no experimental metric bug found
- abstain / no-effect / backfire are not counted as delayed credit success

Limitations:

- hidden indirect discovery still depends on synthetic outcome-history / exploration cues
- high aggregate score is not general structural necessity proof
- real causal discovery is not proven
- real risk intelligence is not proven

## 3. What Has Actually Been Demonstrated

The B-line evidence supports the following conservative claims:

1. In the toy PLOS setting, the system can extend from closed-loop inspection/intervention into risk-constrained actionability.
2. Under an explicit actionability mask, clean risk-constrained policy can choose inspect, direct intervention, indirect intervention, or abstain.
3. Reviewer-hardening stress can expose weaknesses hidden by clean masks and clean risk labels.
4. Missing-mask fallback can recover part of utility in a toy setting.
5. Delayed credit can be diagnosed in synthetic delay5 outcome-history conditions.
6. Trace, candidate search, risk cue, and inspection recovery ablations cause split-level drops.
7. wrong_trace_no_public_state reduces the state_only shortcut.
8. Feedback, history, and credit-buffer necessity can be tested more sharply in targeted synthetic splits.
9. Hidden indirect discovery can work in synthetic exploration/outcome-history diagnostics, but this does not generalize to real causal discovery.

## 4. What Has Not Been Demonstrated

The B-line evidence does not demonstrate:

1. robust trace repair solved
2. general private trace necessity
3. general feedback/history necessity
4. general delayed credit solved
5. hidden indirect causal path discovery in realistic environments
6. autonomous target discovery
7. real-world risk intelligence
8. robot control
9. construction-site autonomy
10. safety certification
11. engineering deployment readiness
12. language-free cognition solved

## 5. Core Remaining Scientific Risks

1. Public state cues may still explain too much.
2. Mask and candidate search may still encode operational shortcuts.
3. Synthetic outcome-history cues may be too clean.
4. Targeted splits may overfit diagnostics.
5. High aggregate score may hide unresolved generalization failures.
6. Trace repair may remain scenario-dependent.
7. Hidden indirect discovery may be synthetic rather than causal.
8. Mechanism necessity is split-specific, not general.

## 6. Strongest Next Question

The strongest next question is:

```text
Do these operational structures survive transfer?
```

The current evidence chain shows increasingly strong toy diagnostics under intervention pressure, risk constraints, delayed outcomes, trace uncertainty, and ablations. It does not yet show whether the structure transfers when surface cues, visual mappings, risk encodings, delay profiles, or dynamics are remapped.

This is the next scientifically meaningful bottleneck.

## 7. Recommended Next Stage

Recommended next stage:

```text
B6.4 Transfer and Anti-Overfit Generalization
```

or:

```text
B6.4 Cross-Environment Operational Structure Transfer
```

B6.4 should not add real-world control, 3D, robotics, or engineering deployment. It should remain toy-to-toy transfer.

Candidate B6.4 tasks:

1. new toy dynamics with the same latent operational structure
2. visual remapping
3. risk cue remapping
4. delayed effect remapping
5. hidden indirect path remapping
6. train/config on original PLOS, evaluate on remapped PLOS
7. compare:
   - b63_1_policy
   - state_only
   - mask_only
   - trace_only
   - random
   - always_abstain
   - oracle

Core question:

If the structure is real, it should partially transfer after surface cues change. If performance collapses, the current mechanism is likely split-specific rule fitting.

## 8. Claim Boundary

The B-line evidence supports a progressively stronger toy diagnostic chain for operational structure under intervention pressure.

It does not yet establish a general theory of language-free cognition.
It does not establish real-world safety intelligence.
It does not establish robotics readiness.
It does not establish construction-site deployment readiness.

## 9. Branch / PR Ledger

- B6.2 branch:
  - branch: b6-2-fallback-delayed-credit
  - commit: c42af92

- B6.3 branch:
  - branch: b6-3-structural-necessity
  - PR: #2
  - latest commit: 60b89d7

- B6.3.1 branch:
  - branch: b6-3-1-wrong-trace-mechanism-necessity
  - commit: 0ffa312
  - PR create URL: https://github.com/chleya/relation-internalization-program/pull/new/b6-3-1-wrong-trace-mechanism-necessity

