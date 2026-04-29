# Relation Internalization Test

## Goal

Test whether a system has internalized external relations rather than merely fitting input-output patterns.

## Core idea

Internalization means external relations become internal usable structures: they can transfer under OOD combinations, adapt under relation reversal, answer counterfactual changes, and be edited with localized behavioral effects.

## Tests

- Relation reversal
- Frozen reversal
- Edit-based reversal
- OOD transfer
- Adversarial spurious-correlation attack
- Counterfactual resource intervention
- Internal editing
- Resource-level internal editing
- Internal relation shuffle audit
- Ground-truth relation table alignment audit

## Run

```powershell
pip install -r requirements.txt
python -m src.run_sweep --config configs/sweep.yaml
python -m src.visualize --summary results/summary.csv
pytest -q
```

Run one model:

```powershell
python -m src.run_experiment --config configs/base.yaml --agent relation --seed 0
```

## Expected Output

```text
results/summary.csv
figures/reversal_adaptation.png
figures/ood_success.png
figures/edit_score.png
figures/internalization_score.png
reports/auto_report.md
```

## Agents

- `majority`: learns one global action.
- `memory`: memorizes full contexts.
- `fitting`: black-box action classifier.
- `predictive`: predicts resource then chooses an action.
- `decision_tree`: interpretable tree baseline over one-hot features.
- `relation`: maintains editable condition -> resource rules.
- `wide_relation`: same relation table mechanism, but it must choose among all feature pairs, including spurious `color/odor`.
- `robust_wide_relation`: wide relation model with an explicit nuisance-aware rule-selection penalty for `color/odor`.

## Claim Boundary

This is a minimal testbed. Passing it is evidence that the model uses explicit relation-like structures in this environment. It is not evidence for a broad intelligence theory.

## v0.2 Hard Setting

The spurious `color` and `odor` features now use larger token vocabularies. Training samples use only a small correlated subset, while OOD samples use held-out tokens. This makes memorizing full contexts and fitting spurious correlations less reliable.

## v0.3 Reversal Guard

In reversal tests, `texture/wet` determine the reversed resource, but `color/odor` remain correlated with the old base world. This prevents spurious features from leaking the new label and makes reversal a real relation-change test.

## v0.4 Spurious Attack

The evaluation now includes adversarial contexts where the surface cues are intentionally flipped: food-like `texture/wet` can carry poison-correlated `color/odor`, and poison-like `texture/wet` can carry food-correlated `color/odor`. A relation-internalized model should follow `texture/wet`, not the flipped surface cues.

## v0.5 Internal Relation Audit

The relation model now supports a shuffle audit. After training, supported high-specificity rule outcomes are shuffled. If behavior still works after this corruption, the model is not actually relying on its internal relation table.

## v0.6 Relation Table Alignment

The report now audits the explicit relation table against the six ground-truth `texture/wet -> resource` rules. This checks whether the model exposes the right relations, not just whether its actions are correct.

## v0.7 Resource-Level Edit Audit

Internal edit evaluation now checks both action changes and inferred resource changes. This matters because `poison` and `neutral` both map to `avoid`, so action-only edit success can hide an incorrect internal relation.

## v0.8 Fairness Check

The sweep now includes `wide_relation` and `decision_tree`. `wide_relation` is a stricter fairness check: it is not handed only `texture/wet`; it enumerates all feature pairs and must survive spurious `color/odor` pressure. `decision_tree` is a strong interpretable baseline.

## v0.9 Nuisance-Aware Relation Selection

`robust_wide_relation` keeps the wide candidate space but downweights rules containing declared nuisance features (`color`, `odor`). This is not spontaneous causal discovery; it tests whether an explicit anti-spurious selection prior can recover robust behavior while preserving relation-table auditability.

## v1.0 Gated Evaluation

The project now reports `gated_internalization_score`, inspired by SVT-style gated evaluation. If any critical gate fails, the gated score is zero. Gates include OOD success, spurious resource accuracy, counterfactual accuracy, resource-level edit, edit-based reversal, relation-table alignment, and relation-table shuffle drop.
