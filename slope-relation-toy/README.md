# Slope Relation Toy

Minimal slope-engineering relation internalization toy world.

Goal:

```text
Move the relation-internalization test from food-world to a slope-engineering relation chain.
```

Core chain:

```text
Rainfall -> Infiltration -> PorePressure -> Displacement -> CrackExpansion -> Risk
Drainage -> PorePressureDown
Anchoring -> DisplacementDown
ToeExcavation -> StabilityDown
Monitoring -> UncertaintyDown
StopWork -> ExposureRiskDown
```

Run:

```bash
pip install -r requirements.txt
python -m src.run_experiment --seeds 0 1 2 3 4
python -m src.visualize --summary results/summary.csv
python -m src.export_review_example
python -m src.self_audit --summary results/summary.csv --output reports/self_audit.md
pytest -q
```

Expected outputs:

```text
results/summary.csv
figures/slope_scores.png
reports/auto_report.md
reports/review_example.json
reports/self_audit.md
```

Baselines include:

```text
surface: learns warning labels rather than relation chains
structural_memory: memorizes structural contexts without editable relation chains
learned_links: learns a small editable relation-link table from observations
generic_review: says generic drainage/support words without a concrete relation chain
relation_chain: explicit auditable relation-chain model
```
