# Neural Relation Internalization under Edit Pressure

This project tests whether neural relation internalization can emerge under edit pressure, rather than relying on hand-written relation agents.

The environment is a toy food/poison relation world. It is not evidence of general causal discovery or large-model relation understanding.

## Models

- `pure_prediction`
- `prediction_bottleneck`
- `counterfactual_training`
- `edit_pressure_training`
- `explicit_table_oracle`

## Run

```bash
pip install -r requirements.txt
pytest -q
python -m src.run_sweep --config configs/sweep.yaml
python -m src.visualize --summary results/summary.csv
python -m src.run_failure_localization --config configs/sweep.yaml
```

Single model:

```bash
python -m src.run_experiment --config configs/base.yaml --model edit_pressure_training --seed 0
```

## Outputs

```text
results/summary.csv
results/records.csv
results/extracted_tables/*.json
figures/gated_scores.png
figures/ood_shortcut.png
figures/reversal_adaptation.png
figures/table_extraction.png
figures/subspace_drops.png
reports/AUTO_REPORT.md
reports/SELF_AUDIT.md
reports/EDIT_PRESSURE_FAILURE_ANALYSIS.md
results/gate_failure_matrix.csv
results/subspace_drop_by_seed.csv
results/intervention_site_analysis.csv
results/extraction_subspace_correlation.csv
```

## Claim Boundary

This is a toy diagnostic for neural relation internalization under edit pressure.

It does not claim:

- real relation understanding;
- general causal discovery;
- large-model behavior;
- deployment-ready engineering AI.
