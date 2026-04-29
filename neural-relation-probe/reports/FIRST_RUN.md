# First Runs

Date: 2026-04-28

## v0.1 Command

```bash
python -m src.run_probe --seed 0 --output results/metrics_seed0.json
```

## v0.1 Observed Result

```text
train_accuracy: 1.0
test_accuracy: 1.0
ood_accuracy: 1.0
spurious_attack_accuracy: 1.0
relation_probe_accuracy: 1.0
random_label_probe_accuracy: 0.214
probe_selectivity: 0.786
relation_dimension_zero_drop: 0.0067
random_dimension_zero_drop: 0.0
gated_neural_relation_score: 0.0
```

## v0.1 Interpretation

The hidden state contains readable relation information, but zeroing the top probe dimensions barely changes task behavior. Under the gate logic, this is not enough to claim neural relation internalization.

This is useful because it prevents a false positive:

```text
readable representation != causally used representation
```

## v0.2 Update

Added:

- linear relation-subspace removal;
- nuisance `color/odor` subspace control;
- 5-seed sweep;
- summary figure.

Command:

```bash
python -m src.run_sweep --seeds 0 1 2 3 4 --output results/summary.csv
python -m src.visualize --summary results/summary.csv
```

Mean over seeds 0-4:

```text
test_accuracy: 1.000
ood_accuracy: 1.000
spurious_attack_accuracy: 1.000
relation_probe_accuracy: 1.000
random_label_probe_accuracy: 0.207
probe_selectivity: 0.793
relation_dimension_zero_drop: 0.013
random_dimension_zero_drop: 0.000
relation_subspace_drop: 0.483
nuisance_subspace_drop: 0.000
gated_neural_relation_score: 0.819
```

Interpretation:

Dimension zeroing was the wrong intervention. Linear relation-subspace removal shows a stable causal effect, while nuisance-subspace removal does not.

This supports a narrow claim:

```text
In this tiny MLP, relation information is linearly readable and behaviorally relevant under subspace removal.
```

It still does not support a broad claim about spontaneous relation discovery.
