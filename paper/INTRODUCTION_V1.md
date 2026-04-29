# Introduction V1

Agents can succeed on relational tasks for reasons that do not justify saying they have internalized the relation. A classifier may predict the right label by exploiting a correlated surface feature. A policy may act successfully by memorizing contexts. A temporal model may predict delayed outcomes without representing editable delay links. A safety-oriented agent may appear conservative by inspecting every missing field. A neural model may respond to an explicit edit signal without binding its query behavior to relations inferred from support examples.

These cases are not edge failures. They are the central problem for evaluating relation-internalization claims. If the evidence is only task success, prediction accuracy, probe readability, plausible explanation text, or edit responsiveness, then many false positives will pass.

This paper studies relation internalization as an operational diagnostic standard in controlled toy environments. We do not propose a general theory of intelligence, and we do not claim real engineering safety or deployment readiness. Instead, we ask a narrower question:

```text
What must a relation-like structure be usable for before it should count under a relation-internalization diagnostic?
```

Our answer is that relation structure must be usable for transfer, counterfactual action, local edits, audits, temporal indexing, uncertainty recognition, and cost-aware inspection. These requirements are implemented as gated diagnostics. A model can be accurate, useful, or interesting while failing a gate; the point is that it should not receive the stronger relation-internalization claim under this diagnostic.

The paper is organized as a false-positive elimination ladder. Each stage introduces a baseline that should pass if ordinary performance were enough, then shows that the baseline fails once structural relation use is required. The ladder includes prediction-only neural models, bottleneck compression, probe-readable shortcut models, structural memory, generic review text, temporal memory, fixed-delay templates, relation discovery without uncertainty, blanket inspection, simple inspection heuristics, and edit-signal responsiveness without support-conditioned binding.

Several positive agents in the R-series and slope/temporal diagnostics are intentionally hand-designed. They are not evidence of spontaneous relation emergence. They are methodological controls: they define what usable, editable, auditable relation structure would need to support, and they expose which weaker baselines should not be credited. To reduce the hand-design concern, we include a neural stage without a hand-written positive relation table. In that stage, pure prediction and bottleneck compression fail the gates, counterfactual training is the strongest current non-handwritten positive condition, and edit-pressure training is mixed.

The edit-pressure result is especially important. It prevents a tempting overclaim. The model can support table-level editability and respond strongly to edit-state swaps, but V1.1 and V1.2 show weak support-conditioned binding and unstable causal relation-subspace evidence. Thus, editable behavior itself becomes another false positive: edit responsiveness is not sufficient unless paired with support-conditioned relation use and causal representation evidence.

The contribution of this work is therefore not a new large model or a claim that relations have been solved. It is a staged diagnostic methodology for asking stronger questions before crediting relation internalization.

## Contributions

1. We propose an operational diagnostic standard for relation internalization in toy environments, requiring transfer, counterfactual use, edits, audits, temporal indexing, uncertainty recognition, and cost-aware inspection.

2. We build a false-positive elimination ladder showing that ordinary success metrics can be high while gated relation-internalization scores are zero.

3. We connect explicit relation-agent diagnostics with neural diagnostics. The current neural stage shows that counterfactual training is the strongest non-handwritten positive condition, while edit-pressure exposes editability as a false positive when support-conditioned binding is weak.

4. We provide reviewer-facing claim boundaries: the work is a diagnostic methodology, not a real slope monitoring system, not general causal discovery, not evidence about large-model relation behavior, and not a deployment claim.
