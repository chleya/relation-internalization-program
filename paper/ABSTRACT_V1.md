# Abstract V1

When should an agent be credited with relation internalization rather than merely predicting outcomes, memorizing contexts, following shortcuts, producing plausible review text, or responding to edit signals? We study this question as a staged diagnostic methodology in controlled toy environments. Each stage introduces false-positive baselines that can look strong under ordinary metrics but fail structural gates.

The diagnostic ladder covers neural prediction and probing, explicit relation chains, temporal delay, partial observability, relation-specific uncertainty, and active inspection under cost. The results show that task success, prediction accuracy, bottleneck compression, probe readability, structural memory, generic review text, temporal prediction, blanket inspection, simple inspection heuristics, and edit-signal responsiveness are insufficient evidence. The strongest current non-handwritten neural positive result is counterfactual training, while edit-pressure training is mixed: it supports table-level editability and edit-state responsiveness, but not robust support-conditioned relation binding or stable causal relation subspaces.

The supported claim is narrow. In these toy diagnostics, relation-internalization claims require relation structures that are usable for transfer, counterfactual action, local edits, audits, uncertainty recognition, and cost-aware inspection. The work is a diagnostic methodology, not a claim about real engineering safety, general causal discovery, or large-model behavior.

## Short Abstract

We present a staged toy diagnostic methodology for evaluating relation-internalization claims. Across neural, temporal, partial-observability, and active-inspection settings, we show that prediction accuracy, probe readability, structural memory, review-like explanations, temporal prediction, blanket inspection, and edit-signal responsiveness can all be false positives. The strongest current non-handwritten neural positive result is counterfactual training; edit-pressure remains mixed, exposing editability itself as a false positive unless paired with support-conditioned binding and causal representation evidence.
