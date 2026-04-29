# V6 Final Report: Multi-Party Audit Resolution

Date: 2026-04-29

## 1. Stage Purpose

V6 tests what happens when multiple plausible review outputs disagree.

The question is not:

```text
Can the system arbitrate real engineering disputes?
```

The question is:

```text
Can the system preserve disagreement, compare relation evidence, retain minority
risk arguments, block automatic resolution, and route the case to human review?
```

## 2. Boundary

V6 is a toy diagnostic.

It does not support:

```text
real engineering arbitration
legal adjudication
expert replacement
deployment governance
real dispute resolution
```

## 3. V6 Base Diagnostic

V6 evaluates toy cases where review outputs disagree on:

```text
status
relation-chain evidence
uncertain links
risk arguments
```

Base gates:

```text
disagreement_detection >= 0.9
evidence_comparison_quality >= 0.8
minority_risk_preservation >= 0.9
no_auto_resolution >= 0.9
human_resolution_routing >= 0.9
audit_trail_completeness >= 0.9
responsibility_boundary >= 0.9
```

Base result:

| resolver | gated_v6_score |
| --- | --- |
| compliant_audit_resolver | 1.000 |
| majority_vote_resolver | 0.000 |
| confidence_only_resolver | 0.000 |
| auto_compromise_resolver | 0.000 |
| ignore_minority_risk_resolver | 0.000 |
| no_audit_trail_resolver | 0.000 |

## 4. V6.1 Hardening

V6.1 attacks:

```text
fake evidence comparison
human-route label without responsibility boundary
hidden auto resolution
disagreement logged but minority risk omitted
tampered resolution hash
```

Hardening result:

| resolver | hardening_v61_gated_score |
| --- | --- |
| compliant_audit_resolver | 1.000 |
| fake_evidence_comparison_resolver | 0.000 |
| human_route_label_only_resolver | 0.000 |
| hidden_auto_resolution_resolver | 0.000 |
| disagreement_logged_no_minority_resolver | 0.000 |
| tampered_resolution_hash_resolver | 0.000 |

Important negative-control result:

```text
hidden_auto_resolution_resolver:
  base_gated_v6_score = 1.000
  hidden_auto_resolution_rejection = 0.000
  hardening_v61_gated_score = 0.000
```

This means a resolver can appear to preserve human routing while secretly
selecting a review outcome. V6.1 catches that false positive.

## 5. Supported Claim

V6 supports this narrow claim:

```text
In toy multi-review cases, audit resolution can be tested for disagreement
detection, relation-evidence comparison, minority-risk preservation, no automatic
resolution, human routing, audit trail completeness, and responsibility boundary.
```

## 6. Unsupported Claims

V6 does not support:

```text
real arbitration
legal adjudication
expert replacement
deployment governance
real engineering approval
```

## 7. Remaining Weaknesses

```text
cases are synthetic
disagreement types are hand-designed
human resolution is only a route label
no real expert process is modeled
resolution hashes are consistency checks, not security
```

## 8. Verification Snapshot

Commands:

```bash
pytest -q
python -m src.run_v6_audit --config configs/v6_audit.yaml
python -m src.run_v61_hardening --config configs/v61_hardening.yaml
```

Current verification:

```text
pytest -q: 11 passed
V6: compliant_audit_resolver = 1.000, base negatives = 0.000
V6.1: compliant_audit_resolver = 1.000, hardening negatives = 0.000
```

## 9. Final Stage Judgment

V6 should be frozen as:

```text
toy multi-party audit-resolution diagnostic
```

Do not interpret it as:

```text
real dispute resolution
real engineering arbitration
legal adjudication
deployment governance
```

