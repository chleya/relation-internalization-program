# B3.1 Active Inspection Degeneracy Audit

## 1. Purpose

B3.1 audits whether identical B3 scores come from model-specific private trace-guided inspection or from shared inspection policy / evaluator degeneracy.

## 2. Background

PLOS v1 found a short-horizon checkpoint candidate. B1.1 showed delayed checkpoint failure. B2 introduced trace-bearing delayed checkpoints. B2.1 hardened trace use. B2.1a found score degeneracy. B2.2 exposed shared selector dependence. B2.3 reconstructed private selectors. B3 showed delayed trace-guided active inspection. B3.1 audits whether B3 active inspection is independently mechanism-separated or still degenerate.

## 3. Audits

- per-episode inspect region overlap
- inspection policy provenance
- trace-family-specific inspection scorer
- shared inspection policy ablation
- disagreement-inspection episodes
- baseline sanity check
- trace-ablation specificity

## 4. Results

| metric | value |
| --- | --- |
| model | cross_model_audit |
| seed | 0 |
| cross_model_inspect_region_match_rate | 1.000 |
| exact_all_model_same_region_rate | 1.000 |
| pairwise_region_match_rate | 1.000 |
| gt_region_match_rate | 1.000 |
| saliency_region_match_rate | 0.000 |
| shared_inspection_policy_usage_rate | 0.000 |
| private_trace_inspection_score_usage_rate | 1.000 |
| fallback_usage_rate | 0.000 |
| unknown_policy_source_rate | 0.000 |
| mean_inspection_scorer_correlation | -0.381 |
| inspection_scorer_specificity | 1.000 |
| shared_policy_ablation_drop | 0.000 |
| private_inspection_retention_after_shared_ablation | 1.000 |
| shared_policy_usage_rate_after_ablation | 0.000 |
| disagreement_inspection_divergence | 1.000 |
| family_aligned_inspection_rate | 1.000 |
| cross_model_same_inspect_rate_on_disagreement | 0.000 |
| causal_family_inspection_accuracy | 0.333 |
| random_inspection_score | 0.006 |
| saliency_inspection_score | 0.000 |
| short_horizon_inspection_score | 0.000 |
| oracle_inspection_score | 1.000 |
| trace_over_saliency_gain_margin | 0.950 |
| trace_over_short_horizon_gain_margin | 0.950 |
| private_trace_ablation_drop | 1.000 |
| matched_non_trace_ablation_drop | 0.000 |
| saliency_ablation_drop | 0.000 |
| private_trace_over_non_trace_ratio | 1000000.000 |
| private_trace_over_saliency_ratio | 1000000.000 |
| non_trace_inspection_stability | 1.000 |
| b31_inspection_audit_score | 0.000 |

## 5. Interpretation

B3.1 does not clear the degeneracy gates; B3 remains evidence for trace-guided inspection behavior, not independent active-inspection mechanisms.

## 6. Claim Boundary

B3.1 does not add a new capability claim.
B3.1 does not prove independent active-inspection mechanisms.
B3.1 only validates or weakens the reliability of B3 active-inspection scoring.
