# Project Charter: Relation Internalization Test

## Project Name

`relation-internalization-test`

## Purpose

Build a minimal, runnable Python experiment to distinguish relation internalization from memory, black-box fitting, and prediction.

## Scope

The project implements one controlled food-poison-texture world, seven agents, hard OOD/frozen/edit reversal tests, adversarial spurious-correlation attacks, internal relation shuffle audits, relation-table alignment audits, plots, a summary CSV, an automatic report, and slope-engineering review templates.

## Non-Goals

- Do not extend the theory.
- Do not add deep learning frameworks.
- Do not claim broad intelligence results.
- Do not optimize for benchmark performance beyond this diagnostic setting.

## Success Criteria

- `pytest -q` passes.
- `python -m src.run_sweep --config configs/sweep.yaml` writes `results/summary.csv`.
- `python -m src.visualize --summary results/summary.csv` writes four figures.
- Relation model is evaluated on online reversal, frozen reversal, edit-based reversal, held-out OOD, counterfactual resource accuracy, and internal edit tests.
- Reversal regimes must not leak the new label through spurious features.
- Spurious attack contexts must flip surface cues against the true `texture/wet` relation.
- Internal relation shuffle should hurt models that claim to rely on explicit relation tables.
- Explicit relation tables should align with the six ground-truth `texture/wet -> resource` rules.
- Internal edits must be evaluated at both action level and resource-inference level.
- Fairness checks must include a wide relation model that is not handed only the causal variables.
- Nuisance-aware relation selection must be reported as an explicit prior, not as spontaneous discovery.

## Required Outputs

- `results/summary.csv`
- `figures/*.png`
- `reports/auto_report.md`
- `engineering/slope_relation_graph.md`
- `engineering/ai_slope_review_template.md`
