# Project Charter: Neural Relation Probe

Date: 2026-04-28

## Purpose

Test whether a small neural model forms and uses internal representations of the `texture/wet -> resource` relation from the food-world task.

This project is the neural follow-up to `F:\relation-internalization-test`.

## Motivation From Existing Projects

### From `F:\新点子文件\svt_agents`

Use gated evaluation. High prediction accuracy is not enough. A model must pass relation-relevant gates:

- OOD gate
- spurious-cue gate
- counterfactual gate
- intervention gate
- identity/relation consistency gate

### From `F:\probe_to_boundary_gnn`

Do not equate probe readability with causal use.

Required separation:

- probe can read relation information from hidden states;
- intervention shows whether the model actually uses that information.

## Minimal Research Question

```text
Does a small neural classifier trained on food-world observations form a hidden representation of the true texture/wet relation, and does its behavior causally depend on that representation?
```

## Non-Goals

- Do not build a large model.
- Do not use LLMs.
- Do not claim general relation discovery.
- Do not migrate to slope engineering until this neural probe is stable.

## Proposed Minimal Setup

Reuse the food-world data generator from `relation-internalization-test`.

Models:

- MLP classifier: context -> resource
- small transformer or residual MLP only after MLP baseline is stable

Hidden probes:

- probe `texture`
- probe `wet`
- probe `texture/wet joint rule`
- probe `resource`
- random-label control

Interventions:

- zero hidden dimensions most predictive of `texture/wet`
- random-dimension zeroing control
- nuisance-dimension zeroing control for `color/odor`

## Required Metrics

- train/test resource accuracy
- held-out OOD accuracy
- spurious attack resource accuracy
- probe accuracy
- random-label probe accuracy
- intervention drop on task accuracy
- random intervention drop
- gated neural relation score

## Gate Logic

The gated score should be zero unless:

- OOD resource accuracy >= 0.8
- spurious attack resource accuracy >= 0.8
- texture/wet probe selectivity >= 0.2 over random-label control
- relation intervention drop > random intervention drop by at least 0.1

## Expected Directory Structure

```text
neural-relation-probe/
  PROJECT_CHARTER.md
  README.md
  requirements.txt
  src/
  tests/
  results/
  figures/
  reports/
```

## First Implementation Step

Implement only:

1. dataset export from food-world contexts;
2. MLP resource classifier;
3. hidden-state extraction;
4. linear probes;
5. random-label control;
6. intervention on top probe dimensions.

No extra theory.
