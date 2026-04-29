# Results Narrative V1

## Overview

The results are best read as a sequence of false-positive eliminations. Each stage contains at least one baseline that can look strong under an ordinary metric but fails when relation structure must be usable for transfer, counterfactual action, edits, audits, uncertainty recognition, or cost-aware inspection.

## Neural Prediction and Probe Results

The neural diagnostics first show that prediction and readability are insufficient. The base neural probe result passes the neural relation diagnostic with a gated score around `0.819`, and the base extracted table reaches `1.000`. However, shortcut-trained neural models can remain probe-readable while failing the gated score. This separates readable relation information from behaviorally specific relation use.

The newer neural training-pressure stage sharpens the conclusion:

| model | gated score | interpretation |
| --- | ---: | --- |
| `pure_prediction` | `0.000` | high ordinary performance is insufficient |
| `prediction_bottleneck` | `0.000` | compression does not isolate relation structure |
| `counterfactual_training` | `~0.981` | strongest non-handwritten neural positive condition |
| `edit_pressure_training` | `~0.200` | mixed; editability false positive |
| `explicit_table_oracle` | `1.000` | hand-written upper bound |

The edit-pressure result is not treated as a success. V1.1 shows unstable relation-subspace evidence across seeds. V1.2 shows strong edit-state responsiveness but weak support-conditioned binding. This identifies editable behavior as a false positive unless paired with support-conditioned relation use and causal representation evidence.

## Engineering-Style Relation Chains

The slope toy asks whether relation chains support action, counterfactuals, edits, audits, and review consistency. `relation_chain` and `learned_links` reach gated scores around `0.98`. In contrast, `structural_memory` and `generic_review` receive gated scores of `0.000` despite ordinary metrics that can look strong.

This stage rules out two common interpretations:

- structural memory is not editable/auditable relation structure;
- plausible review text is not relation use.

## Temporal Relation Diagnostics

Temporal V2/V2.1 show that delayed relation structure is not the same as temporal prediction. `delayed_relation_chain` and `learned_delayed_links` pass temporal and hardening gates with scores of `1.000`. `structural_memory_temporal` and `instant_relation_chain` fail hardening with scores of `0.000`.

This stage rules out:

- same-step relation logic;
- fixed-delay templates;
- temporal memory without editable delay links;
- audit strings without real temporal indexes.

## Active Relation Learning and Discovery

R1, R1.1, and R1.2 move from static relation tables toward active relation agents. R1 `relation_agent` reaches a gated score of `0.972`; R1.1 hardening reaches `1.000`; R1.2 `discovery_relation_agent` reaches `1.000`.

The important role of this line is not to claim spontaneous neural emergence. The agents are explicit controls showing what active relation learning must support: exploration, relation recovery, counterfactuals, edits, hidden-confounder rejection, rule reversal, and candidate discovery.

## Partial Observability

R2 shows that relation discovery alone is insufficient. `discovery_relation_agent` reaches partial observation success of `0.967`, but receives a gated score of `0.000` because it does not reliably handle unverifiable relation chains. `uncertainty_discovery_agent` reaches a gated score of `0.982`.

This is one of the clearest examples for Figure 2:

```text
high partial observation success, zero gated relation score
```

## Relation-Specific and Active Inspection

R2.1 rejects blanket inspection. `missing_always_inspect` receives a gated score of `0.000`, while `relation_specific_uncertainty_agent` reaches `0.933`. R3 then tests field selection under cost and budget. `active_inspection_agent` reaches `0.933`, while random, first-missing, missing-always, and risk-first baselines receive `0.000`.

This final line shows that uncertainty recognition is not enough by itself. The agent must select informative fields, update sequentially, avoid unsafe automation, and avoid overinspection.

## Cross-Stage Result

Across stages, the same pattern repeats:

```text
ordinary success can be high while gated relation score is zero
```

The paper's evidence is cumulative. No single stage proves relation internalization in a broad sense. Instead, each stage removes one false positive that would otherwise make a weaker system look relation-like.

## Suggested Results Section Ending

The final result is a ladder, not a leaderboard. The positive agents show that the gates are satisfiable in the toy settings. The negative controls show why weaker evidence should not receive the stronger relation-internalization claim. The neural stage adds the most important recent qualification: counterfactual training is the strongest current non-handwritten positive condition, while edit-pressure reveals that editable behavior itself can be a false positive.
