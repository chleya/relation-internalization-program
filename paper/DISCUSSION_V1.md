# Discussion V1

The main result is not that a single agent solves relation internalization. The main result is that many signals commonly used as evidence for relation internalization are false positives under stricter diagnostics. The staged design makes these failures visible.

## Prediction False Positive

Prediction accuracy is the weakest evidence. A model can learn the training distribution, exploit shortcut variables, or memorize context-action regularities. In the neural stage, `pure_prediction` has high ordinary performance but receives a gated score of `0.000`. In the broader diagnostic chain, structural memory agents can also look strong under ordinary success metrics while failing edit or audit gates.

The implication is not that prediction is useless. Prediction is necessary for many tasks. It is insufficient for the stronger claim that relation structure has been internalized in a usable way.

## Compression False Positive

Bottleneck compression might appear to encourage abstraction, but compression does not guarantee relation structure. The `prediction_bottleneck` model performs well on several behavioral and extraction metrics, yet relation and nuisance subspace drops are both high. The bottleneck compresses information without cleanly separating relation-specific structure from nuisance structure.

This is a caution against treating compactness as evidence of relation internalization.

## Probe Readability False Positive

Probe readability shows that information is present in a representation. It does not show that the policy uses that information behaviorally. Earlier neural probe results show that shortcut-trained models can retain readable relation information while failing gated relation diagnostics.

The diagnostic therefore requires more than decoding. It asks whether the relation information is behaviorally causal, transferable, and extractable into a table that remains correct under OOD and spurious attack settings.

## Editable Behavior False Positive

The edit-pressure result adds a newer and more subtle false positive. A model can respond to explicit edit signals and support table-level edits without robust support-conditioned relation internalization.

V1.2 makes this distinction concrete:

- `edit_state_swap_success = 1.000`;
- `support_shuffle_drop = 0.000`;
- `support_conditioned_accuracy = 0.500`;
- `binding_sensitivity = 0.000`.

This means the edit pathway has behavioral effect, but query behavior is not reliably bound to support-inferred relations. V1.1 also shows that `edit_pressure_training` fails the relation-subspace gate in four out of five seeds.

The conclusion is conservative: editability is diagnostically useful, but not sufficient. It must be paired with support-conditioned binding and causal representation evidence.

## Review Text False Positive

The slope toy includes generic review baselines because plausible language can create the appearance of relational competence. A system can produce text that sounds like a relation audit while failing relation behavior, edit consistency, or irrelevant-link rejection.

This matters for any system that explains its decisions. The audit must be tied to the relation structure used for action, not merely to plausible post hoc language.

## Temporal Memory False Positive

Temporal prediction can succeed without editable delayed links. Temporal V2/V2.1 separate delayed relation structure from same-step logic, fixed-delay templates, and structural temporal memory. The positive delayed-link agents pass delay edit and temporal audit gates; structural temporal memory and instant relation controls fail hardening.

The broader lesson is that time-indexed prediction is not the same as time-indexed relation structure.

## Discovery False Positive

Relation discovery is not sufficient under partial observability. R2 shows this directly: `discovery_relation_agent` reaches high partial observation success, but receives a gated score of `0.000` because it does not reliably recognize unverifiable relation chains and can automate unsafely.

The diagnostic claim is that discovered relations must be paired with uncertainty recognition. Otherwise, relation discovery can become a source of overconfident action.

## Inspection False Positive

Inspection can also become a false positive. A blanket inspection policy may appear safe because it refuses to act under missingness, but it fails under inspection cost and limited budget. R2.1 rejects `missing_always_inspect`; R3 rejects first-missing, random-field, and risk-first heuristics.

The stronger requirement is targeted, budget-aware information gathering. The agent must inspect fields because they have relation value, not simply because some field is unknown.

## Why the Mixed Edit-Pressure Result Strengthens the Paper

The edit-pressure result would weaken the paper only if the paper claimed that edit pressure is sufficient. It does not. The result strengthens the diagnostic ladder by showing that even a seemingly stronger behavior, editable response, can be a false positive.

This keeps the paper honest. The strongest current non-handwritten neural result is `counterfactual_training`, while `edit_pressure_training` remains a mixed case that reveals a gap between table-level editability, edit-state responsiveness, support-conditioned binding, and stable causal relation subspaces.

## Methodology Rather Than Theory

The correct positioning is methodology. The paper does not identify the essence of relation internalization, and it does not prove that agents naturally acquire relations. It proposes a way to make relation-internalization claims harder to overstate.

This methodology is useful because it requires negative controls. Every major claim is paired with a false-positive baseline that can look good under an ordinary metric but fails a structural gate.

## Main Takeaway

The diagnostic ladder supports a narrow claim:

```text
In controlled toy environments, relation-internalization claims require relation structures that are usable for transfer, counterfactual action, edits, audits, uncertainty recognition, and cost-aware inspection.
```

The paper should not claim more than this. Its value is that it makes weaker evidence visible as weaker evidence.
