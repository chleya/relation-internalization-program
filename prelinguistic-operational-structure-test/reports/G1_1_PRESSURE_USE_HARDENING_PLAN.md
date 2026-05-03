# G1.1 Pressure-Use Hardening Plan

## Purpose

G1 establishes a minimal generator path, but the selected compact rule does not use all intended pressure channels.

Current selected rule:

* use_feedback = false
* use_compression = false

This means G1 is a valid generator start, but it does not yet demonstrate that feedback or compression pressure is necessary for generated operational structure.

## Core Question

Can the generator be forced into conditions where prediction error, compression, intervention usefulness, feedback, and OOD pressure all matter?

## Proposed Changes

1. feedback-required generator split

Create cases where intervention gain is ambiguous unless feedback history is used.

2. compression-required generator split

Create cases where high action utility conflicts with compact mask structure, requiring compression-aware rule selection.

3. pressure ablations

Evaluate generator variants with:

* no prediction error
* no compression surprise
* no intervention gain
* no feedback success
* no OOD remap pressure

4. feature-family stress

Keep the rule-search family lightweight, but add feature dropout and feature remapping to test whether the generated rule is brittle.

## Metrics

* feedback_pressure_gain
* compression_pressure_gain
* pressure_ablation_drop
* generated_rule_pressure_coverage
* generated_mask_f1_under_feature_dropout
* ood_gain_after_pressure_hardening

## Definition of Done

G1.1 succeeds if at least one targeted split requires feedback or compression pressure and the generator uses those channels without evaluator leakage.

Success does not require a high aggregate score.

## Claim Boundary

G1.1 remains a toy generator diagnostic. It does not prove autonomous cognition, real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.
