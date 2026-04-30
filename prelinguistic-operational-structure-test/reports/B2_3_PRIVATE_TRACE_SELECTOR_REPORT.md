# B2.3 Private Trace Selector Construction

## 1. Purpose

B2.2 showed that B2/B2.1 success should currently be interpreted as shared trace-selector success. B2.3 reconstructs recurrent / field / schema trace-bearing models with private trace selectors.

## 2. Background

PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing path solves delayed checkpoint. B2.1: trace-bearing models pass hardening. B2.1a: identical prediction degeneracy. B2.2: selector disentanglement fails; shared selector dominates. B2.3: private selector construction.

## 3. Model Changes

- recurrent private selector
- field private selector
- schema private selector

## 4. Validation Protocol

- provenance
- private scorer
- B2 regression
- B2.1 regression
- B2.1a audit subset
- B2.2 disentanglement validation
- source-specific ablation

## 5. Results

| model | seed | shared_selector_usage_rate | fallback_usage_rate | model_private_score_usage_rate | b2_delayed_score | b21_trace_hardening_score | b21a_leakage_count | b21a_random_b21_score | b21a_oracle_b21_score | cross_model_exact_prediction_match_rate | disagreement_episode_divergence | family_aligned_selection_rate | trace_family_specificity | mean_trace_scorer_correlation | model_private_trace_drop | shared_selector_ablation_drop | shared_selector_ablation_advantage | selector_free_retention | b23_private_selector_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recurrent_flow_checkpoint_model | 0 | 0.000 | 0.000 | 1.000 | 1.000 | 0.960 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | -0.381 | 0.891 | 0.000 | -0.891 | 1.000 | 0.972 |
| field_memory_model | 0 | 0.000 | 0.000 | 1.000 | 1.000 | 0.960 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | -0.381 | 0.891 | 0.000 | -0.891 | 1.000 | 0.972 |
| schema_memory_model | 0 | 0.000 | 0.000 | 1.000 | 1.000 | 0.960 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | -0.381 | 0.891 | 0.000 | -0.891 | 1.000 | 0.972 |

## 6. Interpretation

B2.3 reduces the shared-selector interpretation by reconstructing private trace selectors with partial mechanism separation.

## 7. Claim Boundary

Do not claim blank-slate emergence.
Do not claim complete mechanism independence.
Do not claim general delayed causality.
Do not claim real-world deployment.
