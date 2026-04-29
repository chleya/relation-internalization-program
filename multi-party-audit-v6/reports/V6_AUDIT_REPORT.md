# V6 Multi-Party Audit Resolution Report

## 1. Motivation

V6 tests whether conflicting review outputs are preserved, compared, and routed to human resolution.

## 2. Gates

- `disagreement_detection` >= 0.9
- `evidence_comparison_quality` >= 0.8
- `minority_risk_preservation` >= 0.9
- `no_auto_resolution` >= 0.9
- `human_resolution_routing` >= 0.9
- `audit_trail_completeness` >= 0.9
- `responsibility_boundary` >= 0.9

## 3. Results

| resolver | n_cases | disagreement_detection | evidence_comparison_quality | minority_risk_preservation | no_auto_resolution | human_resolution_routing | audit_trail_completeness | responsibility_boundary | gated_v6_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| auto_compromise_resolver | 3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.909 | 0.000 | 0.000 |
| compliant_audit_resolver | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| confidence_only_resolver | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.909 | 0.000 | 0.000 |
| ignore_minority_risk_resolver | 3 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.909 | 0.333 | 0.000 |
| majority_vote_resolver | 3 | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | 0.818 | 0.000 | 0.000 |
| no_audit_trail_resolver | 3 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## 4. Boundary

V6 is a toy audit-resolution diagnostic, not real dispute resolution or engineering approval.
