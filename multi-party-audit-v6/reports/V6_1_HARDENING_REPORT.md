# V6.1 Audit Resolution Hardening Report

## 1. Motivation

V6.1 attacks false positives in multi-party audit resolution.

## 2. Attacks

- fake evidence comparison
- human-route label without responsibility boundary
- hidden auto resolution
- disagreement logged but minority risk omitted
- tampered resolution hash

## 3. Gates

- `fake_evidence_rejection` >= 0.9
- `human_route_responsibility_consistency` >= 0.9
- `hidden_auto_resolution_rejection` >= 0.9
- `minority_risk_log_integrity` >= 0.9
- `resolution_hash_integrity` >= 0.9

## 4. Results

| resolver | base_gated_v6_score | fake_evidence_rejection | human_route_responsibility_consistency | hidden_auto_resolution_rejection | minority_risk_log_integrity | resolution_hash_integrity | hardening_v61_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compliant_audit_resolver | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| majority_vote_resolver | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| confidence_only_resolver | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| auto_compromise_resolver | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| ignore_minority_risk_resolver | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| no_audit_trail_resolver | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| fake_evidence_comparison_resolver | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| human_route_label_only_resolver | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| hidden_auto_resolution_resolver | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| disagreement_logged_no_minority_resolver | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| tampered_resolution_hash_resolver | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

## 5. Boundary

V6.1 remains a toy audit-resolution diagnostic, not real arbitration.
