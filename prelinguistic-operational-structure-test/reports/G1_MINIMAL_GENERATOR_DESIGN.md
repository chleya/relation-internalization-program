# G1 Minimal Generator Design

## Research Question

Can a minimal generator induce a useful operational structure component instead of receiving it fully hand-designed?

G1 tests whether a compact rule-search generator can synthesize an actionability/update mask from interaction history under pressure from:

1. prediction error
2. compression constraint
3. intervention usefulness
4. feedback success
5. OOD remapping

## Generator Mechanism

G1 uses lightweight exhaustive search over compact rules.

Each rule maps observed interaction-history features to:

* direct actionability
* indirect actionability
* inspectability
* update weight
* generated trace score

The rule is selected by training objective on observed interaction rewards. It does not read evaluator labels, oracle views, or hand-designed actionability masks.

## What Counts As Generated Structure

The generated structure is the learned compact mask/update rule plus the per-region generated mask.

It is generated because the selected thresholds and feature usage are chosen from interaction outcomes, not copied from a B-line evaluator mask.

## What Remains Hand-Designed

G1 is still heavily scaffolded:

* the feature vocabulary is hand-designed
* the candidate rule family is hand-designed
* the synthetic environment is hand-designed
* the evaluator labels are hand-designed but hidden from the generator

This is a minimal generator experiment, not autonomous structure discovery.

## Metrics

G1 reports:

* generator_score
* action_utility
* mask_f1
* compression_cost
* ood_generalization_score
* gain_over_random
* gain_over_hand_designed
* oracle_gap

## Failure Criteria

G1 fails if:

* the generator does not beat random on held-out OOD remap
* the generator only wins by overfitting clean interaction features
* the oracle gap disappears due to evaluator leakage
* generated masks have low F1 while action utility appears high
* reports overstate the result as cognition or deployment readiness

## Expected Negative Results

Likely negative results include:

* hand-designed baselines remain competitive
* OOD remap exposes brittle threshold choices
* compression pressure lowers mask recall
* generated update weights are useful only in the synthetic feature space

## Hardware-Aware Scale

G1 uses CPU-only exhaustive search over a small threshold grid. The default config uses 32 train episodes, 16 test episodes, 16 OOD episodes, and 24 regions per episode.

If runtime becomes excessive, reduce the grid or episode counts while preserving the command and output format.
