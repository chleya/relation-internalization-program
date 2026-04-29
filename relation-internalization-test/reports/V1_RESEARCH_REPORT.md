# V1 Research Report: Relation Internalization Test

Date: 2026-04-28

## Decision

V1 is a usable minimal diagnostic framework, not a broad theory proof.

The strongest supported claim is:

> In this controlled environment, an explicit editable relation-table model is more robust than memory, black-box fitting, predictive classification, and a decision tree under tests that require relation transfer, counterfactual resource inference, internal editing, edit-based reversal, and relation-table auditability.

The strongest unsupported claim is:

> The system proves spontaneous relation discovery in general models.

That stronger claim is not supported because the main `relation` model is handed the causal candidate variables (`texture`, `wet`).

## Experimental Setup

The environment is a food-poison-texture world.

Observed features:

- `texture`: A/B/C
- `wet`: dry/wet
- `color`: spurious token
- `odor`: spurious token

True resource relation:

- `texture=A AND wet=dry -> food`
- `texture=A AND wet=wet -> poison`
- `texture=B -> poison`
- `texture=C -> neutral`

Training uses correlated surface cues. OOD and spurious attack settings break or invert those cues.

## Agents

- `majority`: global action baseline.
- `memory`: full-context memory baseline.
- `fitting`: online SGD action classifier.
- `predictive`: online SGD resource classifier.
- `decision_tree`: interpretable tree baseline over one-hot features.
- `relation`: editable relation table restricted to `texture/wet` candidates.
- `wide_relation`: editable relation table over all feature pairs, including spurious `color/odor`.

## Main Results

Mean over seeds 0-4:

```text
agent          internalization  ood    spurious_resource  cf     edit_resource  edit_reversal  table_alignment  shuffle_drop
relation       0.888            1.000  1.000              1.000  1.000          1.000          1.000            0.367
wide_relation  0.836            1.000  0.375              0.840  1.000          0.982          1.000            0.450
predictive     0.426            0.898  0.533              0.667  0.000          0.000          0.000            0.000
fitting        0.406            0.866  0.325              0.500  0.000          0.000          0.000            0.000
memory         0.291            0.866  0.500              0.506  0.000          0.000          0.000            0.000
decision_tree  0.248            0.773  0.300              0.400  0.000          0.000          0.000            0.000
```

## Interpretation

### What worked

The restricted `relation` model passes every core diagnostic:

- held-out OOD transfer;
- adversarial spurious-correlation attack;
- counterfactual resource inference;
- action-level internal editing;
- resource-level internal editing;
- edit-based reversal;
- relation-table alignment with the six ground-truth rules;
- relation-table shuffle audit.

The shuffle audit matters because it checks whether the model is actually using the relation table. The `relation` model's resource accuracy drops after rule outcomes are shuffled.

### What the fairness check revealed

`wide_relation` is not handed only the causal variables. It enumerates all feature pairs.

It still exposes the correct `texture/wet -> resource` rules:

- table alignment = 1.0

But it fails the adversarial spurious-correlation attack:

- spurious resource accuracy = 0.375

This is an important negative result. It means:

> Discovering or storing the correct relation is not sufficient. The inference policy must also choose the right relation under spurious pressure.

Therefore V1 supports explicit relation-table robustness only when the candidate relation space is constrained or when the model has an additional relation-selection mechanism.

### What did not work

The decision tree baseline did not become a strong relation-internalization baseline in this setting. It fit the training distribution, but its OOD, counterfactual, and spurious-attack behavior were weak.

The predictive baseline can perform well on ordinary OOD for some seeds, but it lacks:

- internal edit support;
- edit-based reversal;
- explicit relation-table alignment;
- relation shuffle auditability.

## Failure Checks

- **Baselines too weak**: Partially addressed by adding `decision_tree` and `wide_relation`. `wide_relation` is the stronger fairness check.
- **Relation model was handed causal variables**: True for `relation`; explicitly exposed by `wide_relation`.
- **Action-only edit success was insufficient**: Fixed by adding resource-level edit metrics.
- **Spurious feature leakage in reversal**: Fixed in v0.3; reversal surface cues remain tied to the old base world.
- **Memory can pass ordinary OOD by action bias**: Addressed by adding resource-level spurious accuracy and table audits.

## Current Claim Boundary

Supported:

- A relation table can be made editable, auditable, and causally relevant in this environment.
- Relation-table corruption degrades relation-model resource inference.
- Explicit relation-table alignment is measurable.
- A model can expose correct relations while still choosing spurious rules at inference time (`wide_relation`).

Not supported:

- General spontaneous relation discovery.
- Superiority over all symbolic or rule-learning systems.
- Readiness for real engineering AI review without a domain-specific toy world.

## Recommended Next Step

Update after v0.9:

The recommended next step from V1 was tested by adding:

```text
robust_wide_relation = wide relation table + explicit nuisance-aware rule-selection penalty
```

Mean results:

```text
agent                 internalization  spurious_resource  cf     edit_resource  edit_reversal  table_alignment  shuffle_drop
robust_wide_relation  0.913            1.000              1.000  1.000          1.000          1.000            0.632
relation              0.888            1.000              1.000  1.000          1.000          1.000            0.367
wide_relation         0.836            0.375              0.840  1.000          0.982          1.000            0.450
```

Interpretation:

`robust_wide_relation` closes the main fairness gap exposed by `wide_relation`. It keeps the full feature-pair candidate space, preserves table alignment, and recovers spurious-attack robustness.

However, this is not spontaneous causal discovery. It uses an explicit nuisance prior:

```text
color/odor rules are downweighted during inference
```

The revised claim boundary is:

> With a declared nuisance-feature prior, a wide explicit relation-table model can both expose the correct `texture/wet` rules and choose them under spurious pressure.

This is stronger than the restricted `relation` result, but still not a fully unsupervised relation-discovery result.

## Next Step After v0.9

Do not add more food-world metrics.

Next work should be one of:

1. Add a relation-selection mechanism to `wide_relation` and test whether spurious attack improves without hiding causal variables.
2. Build the first slope-engineering toy world using the same evaluation harness.
3. Freeze this as V1 and write a short paper-style method note.

The most valuable next experiment is now option 2:

```text
slope-engineering toy world
```

Success criterion:

- preserves relation-table auditability;
- tests interventions such as drainage, anchoring, excavation, stop-work, and monitoring;
- includes false surface cues such as recent weather labels or noisy risk categories;
- supports edit-based reversal and relation-chain audit.

## Addendum: SVT-Style Gated Score

After reviewing `F:\新点子文件\svt_agents`, the project added `gated_internalization_score`.

Gate thresholds:

```text
ood_success >= 0.9
spurious_resource_accuracy >= 0.8
counterfactual_accuracy >= 0.9
edit_resource_success >= 0.9
edit_reversal_success >= 0.9
relation_table_alignment >= 0.9
relation_shuffle_drop >= 0.25
```

If any gate fails, the score is zero.

Mean gated result after v1.0:

```text
robust_wide_relation: non-zero, passes all gates
relation: non-zero, passes all gates
wide_relation: zero, fails spurious-resource gate
all memory/fitting/predictive/tree baselines: zero
```

This follows the SVT lesson: high average performance is not enough when a critical structural gate fails.

## Addendum: Neural Follow-Up Scaffold

The next project has been scaffolded at:

```text
F:\neural-relation-probe
```

Purpose:

```text
Test whether a small neural model encodes and causally uses the texture/wet relation.
```

The scaffold explicitly adopts the `probe_to_boundary_gnn` distinction:

```text
probe readability != causal use
```

The first implementation should include hidden-state probes, random-label controls, and intervention drops before any claim about neural relation internalization.
