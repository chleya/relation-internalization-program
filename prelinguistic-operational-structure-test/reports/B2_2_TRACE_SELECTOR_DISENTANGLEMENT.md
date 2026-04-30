# B2.2 Trace Selector Disentanglement

## 1. Purpose

B2.1a found that three trace-bearing models produce identical per-episode predictions. B2.2 tests whether B2/B2.1 success comes from independent trace mechanisms or a shared trace selector.

## 2. Background

PLOS v1: flow_checkpoint_model became the first checkpoint candidate. B1.1: flow_checkpoint_model failed delayed checkpoint. B2: trace-bearing substrates solved delayed checkpoint. B2.1: trace-bearing models passed trace hardening. B2.1a: identical predictions revealed selector degeneracy. B2.2: disentangles trace selector provenance.

## 3. Methods

- trace provenance audit
- selector-free variants
- trace disagreement episodes
- source-specific ablation
- independent trace scorers
- shared-selector ablation

## 4. Results

| model | seed | shared_selector_usage_rate | model_private_score_usage_rate | fallback_usage_rate | cross_model_exact_prediction_match_rate | selector_free_b21_score | selector_free_retention | disagreement_episode_divergence | family_aligned_selection_rate | trace_family_specificity | mean_trace_scorer_correlation | model_private_trace_drop | shared_selector_ablation_drop | shared_selector_ablation_advantage | private_trace_retention_after_shared_ablation | b22_disentanglement_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.023 | 0.326 | 1.000 | 0.000 | 1.000 | 0.000 | -1.000 | 1.000 | 0.000 |
| field_memory_model | 0 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.023 | 0.326 | 1.000 | 0.000 | 1.000 | 0.000 | -1.000 | 1.000 | 0.000 |
| schema_memory_model | 0 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.023 | 0.326 | 1.000 | 0.000 | 1.000 | 0.000 | -1.000 | 1.000 | 0.000 |

## 5. Interpretation

B2/B2.1 should currently be interpreted as shared trace-selector success, not independent recurrent/field/schema mechanism validation.

The current base trace-bearing models still report shared selector provenance through delayed_common.select_delayed_region. This is the intended reviewer-facing failure mode: B2.2 separates useful trace-bearing behavior from evidence for independent mechanism families.

## 6. Claim Boundary

Do not claim blank-slate emergence.
Do not claim general delayed causality.
Do not claim human-like trace cognition.
Do not claim real-world deployment.
