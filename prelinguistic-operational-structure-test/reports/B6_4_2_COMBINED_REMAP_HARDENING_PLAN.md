# B6.4.2 Combined-Remap Hardening Plan

## Purpose

B6.4.1 hardens individual transfer remaps and removes shortcut-equivalent hard remaps, but combined_remap_hard remains limited.

The next stage should focus only on the combined hard-remap failure.

## Current Blocker

combined_remap_hard:

- policy = 0.675
- oracle = 1.000
- oracle gap = 0.325

The aggregate B6.4.1 score should not hide this failure.

## Scope

B6.4.2 should not enter B7.

It should not add:

- real robots
- real construction-site deployment
- safety certification
- deployable engineering control
- 3D
- LLMs
- language tasks

## Target Questions

1. Which component of the combined remap causes failure?
2. Is the failure driven by visual remap, risk remap, dynamics remap, mask hiding, delay remap, or indirect-path remap?
3. Does the policy fail by abstaining, selecting the wrong indirect path, missing delayed credit, or losing risk inference?
4. Can combined transfer improve without reintroducing shortcut cues?

## Proposed Diagnostics

1. combined_ablation_grid:
   - visual + risk
   - visual + dynamics
   - risk + mask
   - dynamics + delay
   - indirect + mask
   - visual + risk + dynamics
   - visual + risk + dynamics + mask

2. combined_failure_decomposition:
   - classify failure reason per episode
   - separate abstain_uncertain from wrong_action and missed_credit

3. combined_oracle_gap_analysis:
   - measure oracle gap by component combination
   - identify smallest combination that causes collapse

4. shortcut_guard:
   - ensure no public target cue, mask target, trace-only shortcut, or candidate shortcut is reintroduced

## Required Metrics

- combined_component_transfer_score
- combined_component_oracle_gap
- combined_failure_reason_distribution
- combined_abstain_rate
- combined_wrong_action_rate
- combined_missed_credit_rate
- combined_hidden_indirect_failure_rate
- combined_baseline_gap
- shortcut_reintroduction_count

## Definition of Done

B6.4.2 is successful if it explains the combined_remap_hard failure mode and either:

1. improves combined hard transfer without shortcut leakage; or
2. clearly identifies the remaining bottleneck and preserves the failure as a valid limit.

## Claim Boundary

Even if B6.4.2 improves combined remap performance, it supports only toy-to-toy transfer diagnostics.

It does not support real-world transfer, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.
