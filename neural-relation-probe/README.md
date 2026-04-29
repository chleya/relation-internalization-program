# Neural Relation Probe

This is a planned follow-up project to `relation-internalization-test`.

Goal:

```text
Move from explicit relation tables to neural hidden-state tests.
```

The project should reuse the food-world task and test whether a neural model both encodes and uses the true `texture/wet -> resource` relation.

See [PROJECT_CHARTER.md](PROJECT_CHARTER.md).

## Run

```bash
pip install -r requirements.txt
python -m src.run_probe --seed 0 --output results/metrics_seed0.json
python -m src.run_sweep --seeds 0 1 2 3 4 --output results/summary.csv
python -m src.run_modes --seeds 0 1 2 3 4 --output results/summary_modes.csv
python -m src.run_extract --seed 0 --train-mode base --output results/extracted_base_seed0.json
python -m src.run_extract_modes --seeds 0 1 2 3 4 --output results/extraction_summary.csv
python -m src.visualize --summary results/extraction_summary.csv
python -m src.visualize --summary results/summary.csv
pytest -q
```

## Current Interpretation

The first run is expected to be conservative:

```text
probe readability is not enough
```

If the relation probe is accurate but relation intervention does not reduce task accuracy more than a nuisance-control intervention, the gated score remains zero.

Current v0.2 result:

```text
dimension zeroing is too weak;
linear relation-subspace removal produces a stable task drop;
nuisance-subspace removal does not.
```

Current v0.3 result:

```text
base training passes the gates;
shortcut training fails OOD, spurious attack, and relation-vs-nuisance intervention separation.
```

Current v0.4 result:

```text
base neural model -> extracted editable relation table works;
shortcut neural model -> extracted table fails alignment and transfer.
```

Current v0.5 result:

```text
extracted relation tables now have their own gated score;
editable but wrong shortcut tables receive zero.
```
