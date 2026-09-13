# When Prediction Is Not Relation Internalization:
# Editable and Auditable Diagnostics for Relational Agents

# Abstract

When should an agent be credited with relation internalization rather than merely predicting outcomes, memorizing contexts, following shortcuts, producing plausible review text, or responding to explicit edit signals? We study this question as a staged diagnostic methodology in controlled toy environments. Each stage introduces false-positive baselines that can look strong under ordinary metrics but fail relation-internalization gates.

The central thesis is conservative: task success, prediction accuracy, probe readability, memory, surface shortcuts, plausible review text, temporal prediction, blanket inspection, and edit-signal responsiveness are insufficient evidence of relation internalization. The supported claim is narrower: in controlled toy diagnostic environments, relation internalization can be operationalized as relation structures that are usable for transfer, counterfactual action, edits, audits, uncertainty recognition, and cost-aware inspection. The gates are explicit diagnostic requirements for making this claim in the toy settings; they are not proposed as a general theory of intelligence.

# 1. Introduction

Ordinary task accuracy is too weak for claims about relation internalization. A policy can succeed by exploiting shortcut variables, memorizing contexts, using structural memory, producing plausible language, predicting delayed outcomes, inspecting everything, or responding to an explicit edit signal without binding behavior to an inferred relation.

This paper treats relation internalization as an operational diagnostic standard, not as a theory of intelligence. The question is not whether a model "understands" relations. The question is whether relation structure is usable in ways relation claims require: transfer, counterfactual action, intervention/edit, audit, temporal indexing, uncertainty handling, and cost-aware inspection.

Every stage was designed around a baseline that should pass if ordinary performance were enough, but fails once relation internalization is required.

Several positive agents are intentionally hand-designed. They should be read as methodological controls that define what usable relation structure would need to support, not as evidence that arbitrary agents naturally acquire such structure. The neural stage is included to reduce this weakness: it shows that prediction and bottleneck compression fail, counterfactual training is the strongest current non-handwritten positive condition, and edit-pressure remains mixed.

# 2. Operational Definition

We define relation internalization as:

```text
External relations become internal usable structures.
```

Usable means:

- transfer across distribution shift;
- counterfactual use;
- local intervention/edit;
- audit of relation links;
- action guidance;
- temporal indexing when relation effects are delayed;
- uncertainty recognition under partial observability;
- cost-aware inspection under budget.

This standard distinguishes relation internalization from:

- high prediction accuracy;
- bottleneck compression;
- probe readability;
- context memory;
- surface cue use;
- generic review language;
- temporal memory;
- blanket inspection;
- edit-signal responsiveness without support-conditioned binding.

The gates therefore encode claim requirements. A model that fails one gate may still be useful or accurate, but it should not be credited with the stronger relation-internalization claim under this diagnostic.

# 3. False-Positive Ladder

The diagnostic ladder is organized around false positives:

| false positive | why it can look good | why it is insufficient |
| --- | --- | --- |
| prediction success | high train/OOD accuracy | may rely on shortcuts or memorized mappings |
| bottleneck compression | compact representations | can entangle relation and nuisance factors |
| probe readability | relation labels linearly decodable | readable information may not be behaviorally causal |
| structural memory | high success on known patterns | lacks editable/auditable relation links |
| generic review text | plausible explanation strings | language can be detached from relation behavior |
| temporal memory | delayed predictions succeed | lacks editable/auditable temporal links |
| fixed-delay template | passes one delay setting | fails variable delay and delay-edit tests |
| relation discovery without uncertainty | high action success under partial inputs | can automate unsafely when the chain is unverifiable |
| blanket inspection | appears conservative | fails precision, cost, and budget constraints |
| first-missing/random/risk-first inspection | simple heuristics can sometimes work | do not target highest relation value under budget |
| edit-signal responsiveness | edit swaps change outputs | may not bind query behavior to support-inferred relations |

# 4. Experimental Groups

## 4.1 Static Relation Diagnostics

The static line includes the food-world explicit relation table and the R1/R1.1/R1.2 active relation agent line. These tests require transfer, OOD action, relation recovery, counterfactual accuracy, edit success, active exploration, candidate discovery, and rejection of hidden confounders or irrelevant candidates.

Purpose: rule out memory, fitting, shortcut policies, no-exploration relation tables, and predefined `TRUE_LINK` dependence.

These positive agents are not presented as spontaneous neural emergence. They are explicit controls used to define and stress-test relation-usable behavior.

## 4.2 Neural Relation Diagnostics

The neural line includes the hidden-state relation probe, causal subspace intervention, neural-to-table extraction, and the newer counterfactual/edit-pressure stage.

The current neural stage compares:

- `pure_prediction`;
- `prediction_bottleneck`;
- `counterfactual_training`;
- `edit_pressure_training`;
- `explicit_table_oracle`.

The main update is that `counterfactual_training` is the strongest current non-handwritten neural positive result. `edit_pressure_training` is mixed: it shows table-level editability and edit-state responsiveness, but not strong support-conditioned relation binding or stable causal relation subspaces. Therefore, editable behavior itself must be treated as a potential false positive.

This result is deliberately not framed as "edit pressure succeeds." The stronger conclusion is that counterfactual pressure passes the current neural diagnostic more reliably, while edit pressure exposes another way a model can look relation-like without satisfying the full relation-internalization claim.

## 4.3 Engineering-Style Relation Chains

The slope-relation toy evaluates relation-chain agents against `structural_memory`, `generic_review`, `surface`, and other baselines. It tests OOD transfer, spurious attack robustness, counterfactuals, edits, relation audit, irrelevant-link rejection, noisy observations, and review consistency.

Purpose: rule out structural memory, surface labels, and plausible review language as sufficient evidence.

## 4.4 Temporal Relation Diagnostics

Temporal V2 and V2.1 test delayed relation chains. V2.1 hardens the setting with variable delays, false delay shortcut rejection, multi-link delay edits, temporal audit consistency, and anti-template generalization.

Purpose: rule out same-step logic, fixed-delay templates, temporal memory without editable delay links, false temporal shortcuts, and audit strings without real time indexes.

## 4.5 Partial Observability and Active Inspection

R2 tests partial observability; R2.1 tests relation-specific uncertainty versus blanket inspection; R3 tests active inspection selection under cost and budget. The required behavior is not merely to inspect, but to select informative fields, update sequentially, avoid unsafe automation, and avoid overinspection.

Purpose: rule out relation discovery without uncertainty handling, blanket inspection, first-missing/random/risk-first heuristics, and unsafe automation under unverifiable relation chains.

## 4.6 Pre-Linguistic Operational Structure Generation

The G-line tests whether a compact rule-search generator can induce operational structure (actionability mask, update mask, trace revision rule) from interaction history alone, without being given explicit trace selectors, actionability masks, relation tables, or oracle values. This is the generator route (G-line), distinct from the B-line discriminator route where these structures are hand-provided and tested for usage.

G1 searches 2,304 compact rule candidates over direct/indirect/inspect/risk thresholds and update weights. G1.1 adds pressure hardening: the selected rule must use feedback and compression pressure channels under conditions that require them. G1.2 replaces hand-designed scoring formulas with sparse feature induction from a vocabulary of 7 primitive interaction features, searching for the minimal feature subset that maximizes generator score.

Purpose: test whether minimal operational structure can be generated without hand-designed trace selectors or explicit actionability masks.

### G1 Results

| metric | value |
| --- | ---: |
| g1_generator_mean_score | 0.890 |
| g1_ood_score | 0.868 |
| g1_ood_gain_over_random | 0.637 |
| g1_ood_gain_over_hand_designed | 0.273 |
| g1_oracle_gap | 0.110 |
| g1_mask_f1 | 0.934 |

Selected rule: all thresholds at minimum (0.45), risk threshold at maximum (0.62), feedback and compression both disabled. Adversarial review confirms no forbidden evaluator/oracle references in the generator source, but notes that the selected rule does not use feedback or compression pressure channels.

### G1.1 Pressure Hardening Results

| metric | value |
| --- | ---: |
| g1_1_mean_score | 0.597 |
| g1_1_ood_pressure_score | 0.779 |
| feedback_pressure_gain | 0.024 |
| compression_pressure_gain | 0.063 |
| g1_1_oracle_gap | 0.302 |
| g1_1_mask_f1 | 0.533 |

Selected rule: same thresholds as G1 but with feedback and compression both enabled (pressure coverage = 2). Degeneracy audit exposes a critical weakness: on `feedback_required` episodes, the `no_compression` ablation (0.264) outperforms the full generator (0.197); on `compression_required` episodes, the `no_feedback` ablation (0.646) outperforms the full generator (0.633). The pressure channels are not demonstrated to be necessary.

### G1.2 Feature Induction Results

| metric | value |
| --- | ---: |
| g1_2_mean_score | 0.813 |
| g1_2_ood_score | 0.887 |
| feedback_feature_drop | 0.198 |
| compression_feature_drop | 0.308 |
| gain_over_random_feature_program | 0.611 |
| oracle_gap | 0.085 |
| mask_f1 | 0.750 |

Selected program (complexity = 5): direct=(feedback_success, threshold 0.45), indirect=(compression_surprise, threshold 0.65), inspect=(compression_surprise, threshold 0.45), risk_threshold=0.62. Uses 2 of 7 available primitive features.

Feature ablation shows clear condition-specific necessity evidence: on `feedback_required` episodes, dropping the `feedback_success` feature causes a 0.530 drop (0.763 → 0.233); on `compression_required` episodes, dropping the `compression_surprise` feature causes a 0.555 drop (0.708 → 0.153). Oracle gap is 0.085. Adversarial review detects evaluator_ground_truth usage during program selection (a training-time oracle leakage). The feature vocabulary (7 hand-named features) and OOD remap (synthetic noise shift) remain hand-scaffolded.

### G1.2-Clean (Oracle-Free Selection) Results

| metric | value |
| --- | ---: |
| g1_2_clean_mean_score | 0.740 |
| g1_2_clean_ood_score | 0.835 |
| feedback_feature_drop | 0.289 |
| compression_feature_drop | -0.012 |
| oracle_gap | 0.159 |
| mask_f1 | 0.635 |

Training objective uses only observed_reward from interaction_history (no evaluator_ground_truth). Selected program also uses feedback_success and compression_surprise features. Feedback necessity evidence is preserved (drop 0.289), but compression shows near-zero effect (-0.012). The performance drop (0.813 → 0.740) reflects the cost of removing oracle guidance from program selection. This establishes a clean no-oracle baseline for the generator route.

### G1.3 Unsupervised Feature Discovery Results

| metric | value |
| --- | ---: |
| g1_3_mean_score | 0.276 |
| g1_3_ood_score | 0.267 |
| oracle_gap | 0.622 |
| mask_f1 | 0.079 |

Features are discovered via k-means clustering (k=5) on 10-dimensional raw interaction vectors, replacing the 7 hand-named features. The resulting cluster features (e.g., `c1: compression_surprise+delay_signal`, `c2: prediction_error+intervention_gain`, `c3: risk_proxy+compression_surprise`) fail to produce actionable programs. The 0.276 score is only marginally above random (0.201) and the mask F1 of 0.079 indicates near-complete failure to recover structural information from passive clustering. This negative result suggests that passive feature discovery from static interaction data is insufficient; active/interaction-based feature construction (where features are shaped by acting and observing consequences) may be required.

### G2 Compositional World Results

| metric | value |
| --- | ---: |
| g2_mean_score | 0.457 |
| g2_ood_score | 0.335 |
| g2_ood_novel_object_score | 0.506 |
| g2_ood_topology_shift_score | 0.606 |
| object_discovery_ari | 0.610 |
| edge_discovery_f1 | 1.000 |
| indirect_ablation_drop | 0.000 |
| oracle_gap | 0.193 |
| mask_f1 | 0.186 |

G2 scales the generator to a multi-object compositional world with 5 object types (mechanical/thermal/diffusion/advection/collision), cross-object causal edges with propagation delays, and 5 evaluation conditions including `ood_novel_object` and `ood_topology_shift`. Rules are generated via k-means adaptive grouping over 10 interaction dimensions, producing cross-object similarity matrices and actionability masks. The best rule (k=4, similarity_threshold=0.75, priority=lowest_risk) achieves g2_mean_score=0.457, with object_discovery_ari=0.610 and edge_discovery_f1=1.000. However, indirect_ablation_drop=0.000: the `g2_no_indirect` baseline (same rule with `use_indirect=False`) scores identically to `g2_cross_object`. The cross-object indirect pathway is not demonstrated to be necessary. OOD novel object (0.506) and topology shift (0.606) conditions score higher than noise remap (0.335), suggesting the structure is not purely noise-memorized but may rely on statistical shortcuts that generalize across condition types.

### G2.1 Indirect Pathway Pressure Hardening Results

| metric | value |
| --- | ---: |
| g2_mean_score | 0.461 |
| g2_ood_score | 0.335 |
| g2_ood_novel_object_score | 0.506 |
| g2_ood_topology_shift_score | 0.628 |
| indirect_ablation_drop | 0.005 |
| indirect_kill_drop | 0.005 |
| oracle_gap | 0.189 |
| object_discovery_ari | 0.610 |

G2.1 hardens the indirect pathway by two complementary interventions: (1) the rule search space is restricted to only those candidates with `use_indirect=True` (81 rules vs the baseline 162), forcing the generator to use the cross-object indirect channel; (2) a new `g2_kill_indirect` policy nullifies both `similar_groups` and `indirectly_intervenable` in the generated mask, testing whether the indirect pathway carries any structural information beyond the binary flag. The result is decisive: hardening produces nearly identical scores to baseline (0.461 vs 0.457, Δ=0.005), and both `no_indirect` and `kill_indirect` ablations show the same negligible drop (0.005). This replicates the G1.1 degeneracy pattern in the cross-object setting: even when forced to use the indirect channel, the generator derives no measurable benefit from it. The cross-object similarity matrix and indirect intervention logic are present in the generated output but are not causally used. Together with G1.1's finding that feedback/compression channels are similarly unused, this establishes a structural pattern: in the current grid-search architecture operating over hand-decoded interaction features, pressure channels are constructed by the system but bypassed during decision-making.

# 5. Results Summary

| group | positive result | key negative controls | result anchor |
| --- | --- | --- | --- |
| Neural probe/extraction | base neural model and base extracted table pass gates | shortcut neural model | base gated neural score about `0.819`; shortcut gated score `0.000`; base extraction `1.000`; shortcut extraction `0.000` |
| Neural counterfactual/edit pressure | `counterfactual_training` | `pure_prediction`, `prediction_bottleneck`, `edit_pressure_training` as mixed false positive | `counterfactual_training` gated `~0.981`; edit-pressure gated `~0.200` |
| Slope toy | `relation_chain` / `learned_links` | `structural_memory`, `generic_review`, `surface` | relation-chain/learned-links gated about `0.98`; structural/generic controls `0.000` |
| Temporal V2/V2.1 | `delayed_relation_chain`, `learned_delayed_links` | `structural_memory_temporal`, `instant_relation_chain` | delayed-link hardening `1.000`; structural temporal and instant controls `0.000` |
| R1/R1.1/R1.2 | active/discovery relation agents | random, shortcut, passive, no-explore, hand-supplied candidate dependence | R1 `0.972`; R1.1 `1.000`; R1.2 discovery `1.000` |
| R2/R2.1/R3 | uncertainty and active inspection agents | discovery-only, missing-always, first-missing, random, risk-first | R2 uncertainty `0.982`; R2.1 relation-specific `0.933`; R3 active inspection `0.933`; baselines `0.000` |
| G1/G1.1/G1.2/G1.2-Clean/G1.3 | generated operational structure | random, always_abstain, hand_designed; no_feedback, no_compression ablations | G1 `0.890` (OOD `0.868`); G1.1 `0.597` (OOD `0.779`); G1.2 `0.813` (OOD `0.887`, oracle leak); G1.2-Clean `0.740` (OOD `0.835`, no oracle); G1.3 `0.276` (OOD `0.267`, failed discovery) |
| G2/G2.1 | compositional world generator, indirect pathway hardening | random_group, no_indirect, kill_indirect, oracle | G2 `0.457` (OOD `0.335`, ARI `0.610`, indirect drop `0.000`); G2.1 `0.461` (OOD `0.335`, ARI `0.610`, indirect drop `0.005`, kill drop `0.005`) |

The important neural update is the editability false positive. V1.2 shows `edit_state_swap_success = 1.000`, but `support_shuffle_drop = 0.000`, `support_conditioned_accuracy = 0.500`, and `binding_sensitivity = 0.000`. Edit-signal responsiveness is not enough.

# 6. What Each Stage Rules Out

| Stage | Positive agent/result | Negative control | Ordinary metric that could look good | Gate that fails | Alternative explanation ruled out |
| --- | --- | --- | --- | --- | --- |
| Neural probe | base neural model | shortcut neural model | probe readability | OOD/spurious/subspace gates | readable relation information is not enough |
| Neural training pressure | `counterfactual_training` | `pure_prediction` | train/OOD accuracy | shortcut and relation-subspace gates | prediction success is not enough |
| Neural training pressure | `counterfactual_training` | `prediction_bottleneck` | behavior and extraction metrics | nuisance-subspace gate | compression can entangle factors |
| Neural edit pressure | mixed `edit_pressure_training` | edit-signal responsive behavior | table edit/locality | support binding and relation-subspace diagnostics | editable behavior can be a false positive |
| Slope toy | `relation_chain` / `learned_links` | `structural_memory` | OOD/spurious success | edit/audit/review gates | structural memory is not relation structure |
| Slope toy | relation-chain agents | `generic_review` | plausible text | relation behavior gates | review text is not relation use |
| Temporal V2/V2.1 | delayed-link agents | `structural_memory_temporal` | temporal prediction | delay edit/audit gates | temporal memory is not delayed relation structure |
| Temporal V2/V2.1 | delayed-link agents | `instant_relation_chain` | same-step relation logic | variable delay and edit gates | immediate logic cannot replace temporal indexing |
| R2 | `uncertainty_discovery_agent` | `discovery_relation_agent` | partial observation success `0.967` | uncertainty/unsafe automation gates | discovery alone is not enough |
| R2.1 | relation-specific uncertainty | `missing_always_inspect` | high critical recall | precision/cost gates | blanket inspection is not uncertainty handling |
| R3 | active inspection agent | first/random/risk-first baselines | some inspection success | target/cost/budget gates | simple inspection heuristics are not cost-aware selection |
| G1 | generated rule from interaction history | `hand_designed` baseline | hand-designed formulas | OOD remap; generator must beat hand-designed | hand-designed rules are not necessary for operational structure |
| G1.1 | pressure-hardened generator | `no_feedback` / `no_compression` ablations | pressure-enabled rule | degeneracy audit; ablations must lose score | enabled pressures can be false positives when ablations do not degrade |
| G1.2 | sparse feature-induction program | `no_feedback_feature` / `no_compression_feature` | selected feature combination | feature drop must be condition-specific and large (>0.15) | feature necessity must be demonstrated per condition, not assumed |
| G1.2-Clean | oracle-free observed-reward selection | same ablations as G1.2 | no evaluator_ground_truth in training | feedback drop 0.289 > 0.15 | oracle-leakage-free program selection is viable but loses performance |
| G1.3 | k-means feature discovery (5 clusters) | random_discovered, cluster-ablations | passive clustering of raw vectors | no condition passes 0.30 threshold | passive feature discovery cannot replace hand-named features on this toy |
| G2 | cross-object compositional generator | `g2_random_group`, `g2_no_indirect` | multi-object world with cross edges | indirect_ablation_drop = 0.000; indirect channel is not necessary | cross-object similarity and indirect intervention are unused shortcuts |
| G2.1 | indirect pathway pressure hardening | `g2_kill_indirect` | forced `use_indirect=True` rule search | indirect_kill_drop = 0.005; nullifying similar_groups has near-zero effect | indirect pathway is constructed but never causally used; degeneracy pattern replicates G1.1 |

# 7. Discussion

The paper is best read as a false-positive analysis.

Probe readability is insufficient because a relation can be decodable without being behaviorally causal. Prediction is insufficient because success can come from shortcuts, memory, or structural correlation. Review text is insufficient because an explanation can be plausible while disconnected from relation use. Temporal prediction is insufficient because delayed links must be editable and auditable. Discovery is insufficient because a discovered relation may be unverifiable in the current observation. Inspection is insufficient because blanket conservatism fails under cost and budget.

The neural stage adds a final caution: editable behavior is not necessarily relation internalization. `edit_pressure_training` can respond to edit signals and support table-level edits without robust support-conditioned binding or stable causal relation subspaces. The strongest current neural positive result is therefore `counterfactual_training`, not edit pressure.

The G-line adds a generator-route perspective to the predominantly discriminator-route methodology. G1 shows that compact rule search over interaction history can induce operational structure that beats hand-designed baselines on OOD remap, but the selected rule avoids feedback and compression channels entirely, suggesting a shortcut-like solution. G1.1 exposes this as a degeneracy: when forced to use pressure channels, the ablations that drop these channels sometimes outperform the full generator. G1.2 partially resolves this by using sparse feature induction (2 of 7 features) and producing condition-specific necessity evidence (feedback drop -0.530, compression drop -0.555), but the training-time oracle leakage via `evaluator_ground_truth` during program selection must be treated as a known methodological limitation. G1.2-Clean removes the oracle from program selection, using only observed_reward data from interaction_history as the training objective; the resulting generator scores 0.740 (OOD 0.835) with preserved feedback necessity evidence (drop 0.289), demonstrating that oracle-free generator training is viable but incurs a measurable performance cost. G1.3 attempts unsupervised feature discovery via k-means clustering of raw interaction vectors, replacing all 7 hand-named features; the result (0.276, oracle gap 0.622) represents a definitive failure of passive clustering to recover structural information, suggesting that active/interaction-based feature construction may be required. G2 extends the generator to a multi-object compositional world with cross-object causal edges and propagation delays; the generator achieves g2_mean_score 0.457 with object_discovery_ari 0.610, but the indirect_ablation_drop is 0.000—the cross-object indirect pathway is not necessary. G2.1 hardens this pathway by restricting the rule search to indirect-only candidates and adding a kill_indirect ablation that nullifies similar_groups; the result (indirect_kill_drop 0.005) decisively replicates the G1.1 degeneracy pattern in a different structural channel. The cross-object similarity matrix and indirect intervention logic are constructed but never causally used. Taken together, G1.1's feedback/compression degeneracy and G2.1's indirect pathway degeneracy establish a structural pattern: in the discrete grid-search architecture operating over hand-decoded interaction features, any pressure channel can be constructed by the generator and then bypassed during decision-making. The generator route thus demonstrates a systematic limitation: operational structure can be induced from pressure signals (G1, G1.2-Clean) and oracle leakage can be isolated, but the induced structure does not survive degeneracy audit when any single channel is ablated. Passive feature discovery (G1.3) fails entirely.

# 8. Limitations

- All environments are toy diagnostics.
- Variables are hand-specified.
- Some stages use predefined candidate relation graphs.
- Uncertainty and inspection policies are rule-based.
- Data generators are synthetic.
- The neural setting is small and controlled.
- Counterfactual training may be viewed as supervised shortcut control, not evidence of broad emergence.
- The gates are author-defined diagnostic criteria; future work should test pre-registered or externally generated variants.
- Several positive agents are hand-designed relation-structure controls rather than evidence of natural emergence.
- No real slope mechanics are modeled.
- No real sensor reliability calibration is performed.
- No adversarial missingness robustness is claimed beyond tested toy cases.
- No real engineering safety or deployment claim is made.
- No claim is made about large language models or general causal discovery.
- G1.2 uses `evaluator_ground_truth` during program selection; this is a training-time oracle leakage.
- G1.2-Clean removes oracle leakage but relies on observed_reward proxies which are imperfect correlates of ground-truth labels.
- G1.3's passive k-means clustering fails to recover actionable feature structure; this negative result is documented but may reflect the simplicity of the clustering method rather than an impossibility.
- The G-line feature vocabulary in G1/G1.1/G1.2/G1.2-Clean is hand-named; the system does not discover feature concepts from raw data.
- G2's compositional world uses synthetically generated latent profiles and cross-object edges; the object types and causal structures are hand-designed rather than emergent from continuous dynamics.
- G2.1's hardening restricts the search space from 162 to 81 rules, which may reduce statistical power for detecting small but real indirect pathway effects; the 0.005 drop should be interpreted as an upper bound on the indirect pathway's contribution, not as a precise estimate.
- The systematic degeneracy pattern (G1.1 feedback/compression drop < 0.07, G2.1 indirect/kill drop 0.005) is observed in two distinct pressure channels under the same discrete grid-search architecture; generalization to other channels or architectures is not established.

# 9. Future Work

Near-term:

- Replace rule-based inspection value with learned uncertainty/value estimation while preserving gates.
- Add adversarial missingness and correlated sensor failures.
- Add confidence calibration curves.
- Connect neural-to-table extraction with active inspection.
- Test whether counterfactual-trained neural structures remain editable under larger relation graphs.
- G2.2: test whether generated operational structures transfer to genuinely novel world dynamics (continuous PDE simulation, not synthetic noise remaps or hand-designed latent profiles).
- G2.3: investigate active/interaction-based feature discovery where features are constructed by testing actions and observing outcomes, not by passive clustering or hand-named vocabularies.
- Investigigate neural constructive agent that learns pressure-channel representations end-to-end through gradient-based training rather than discrete grid search.

Medium-term:

- Train neural policies to learn editable/auditable relation structures without hand-written relation agents.
- Learn sensor reliability instead of hand-specifying it.
- Scale to larger variable graphs and more complex temporal dependencies.

Long-term:

- Move from toy worlds to validated simulators.
- Define evidence required before any engineering safety claim.
- Build benchmark families combining distribution shift, intervention, temporal delay, partial observability, and inspection cost.

# 10. Final Claim

We present a staged toy diagnostic framework showing that relation internalization can be separated from prediction, probe readability, memory, surface shortcuts, review-like text, temporal prediction, blanket inspection, and edit-signal responsiveness. In these environments, agents pass diagnostic gates only when relation structure is usable for transfer, counterfactual action, edits, audits, uncertainty recognition, and cost-aware inspection.
