# B2.1a Trace Hardening Score Degeneracy Audit

## 1. Purpose

B2.1 produced identical scores for recurrent, field, and schema trace-bearing models. This audit checks whether that result is genuine or caused by evaluator degeneracy, leakage, fallback behavior, or non-discriminative gates.

## 2. Background

PLOS v1 found flow_checkpoint_model. B1.1 showed flow_checkpoint_model fails delayed checkpoint. B2 introduced trace-bearing substrates. B2.1 showed all three trace-bearing models pass with identical score 0.959. B2.1a audits that identical-score result.

## 3. Audits

- per-attack breakdown
- per-seed breakdown
- predicted region distribution
- ground-truth leakage
- intervention applicability
- random-trace baseline
- oracle baseline
- trace-family ablation
- no-trace ablation

## 4. Results

| metric | value |
| --- | ---: |
| `score_degeneracy_detected` | 1.000 |
| `all_attacks_identical_flag` | 1.000 |
| `exact_same_score_all_models_all_seeds` | 1.000 |
| `cross_model_exact_prediction_match_rate` | 1.000 |
| `gt_region_match_rate` | 0.695 |
| `saliency_region_match_rate` | 0.000 |
| `leakage_count` | 0.000 |
| `intervention_applicability_rate` | 1.000 |
| `fallback_rate` | 0.000 |
| `structure_changed_rate` | 1.000 |
| `output_changed_rate` | 0.969 |
| `random_b21_score` | 0.000 |
| `oracle_b21_score` | 1.000 |
| `trace_family_ablation_drop` | 1.000 |
| `no_trace_ablation_drop` | 1.000 |
| `b21a_degeneracy_audit_score` | 0.000 |

## 5. Interpretation

B2.1 identical scores are not yet reliable evidence for robust trace-bearing substrates.

The audit detects exact cross-model prediction matching and identical per-attack metrics. Because this is not explained by oracle-level true-region agreement across all records, the score remains zero even though leakage, random-baseline, intervention, and ablation checks pass.

The audit also shows no detected ground-truth key leakage, low random baseline score, high oracle score, applicable interventions, and large trace/no-trace ablation drops. The remaining problem is not leakage or random passability; it is that the three models share effectively identical predictions under B2.1.

## 6. Claim Boundary

This audit does not prove blank-slate emergence.
This audit does not prove general delayed causality.
This audit only validates or invalidates the reliability of B2.1 scoring.
