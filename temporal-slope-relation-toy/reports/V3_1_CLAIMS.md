# V3.1 Claims

Date: 2026-04-28

## 1. Stage Claim

V3.1 supports this narrow claim:

```text
In the toy V3 setting, takeover can be hardened against generic overcaution:
an agent should reject irrelevant or benign uncertainty while still taking over
for concrete relation-chain conflicts and providing specific audit.
```

## 2. Supported Claims

### 2.1 Generic Uncertainty Takeover Is Not Enough

V3.1 adds an explicit negative control:

```text
overcautious_takeover
```

This agent takes over whenever any uncertainty appears. It gets:

```text
hardening_v31_gated_score = 0.0
```

This supports:

```text
High takeover recall alone is not sufficient.
```

### 2.2 Relation-Specific Takeover Survives Hardening

Current result:

```text
delayed_relation_chain             1.0
learned_delayed_links              1.0
uncertainty_aware_delayed_links    1.0
```

These agents pass:

```text
irrelevant_missing_rejection
benign_noise_rejection
takeover_overuse_control
conflicting_evidence_takeover
audit_specificity
```

### 2.3 Structural Memory Still Fails

`structural_memory_temporal` rejects some irrelevant uncertainty but fails the core relation-chain takeover and audit requirements:

```text
conflicting_evidence_takeover = 0.0
audit_specificity = 0.0
hardening_v31_gated_score = 0.0
```

This preserves the V3 conclusion:

```text
temporal memory is not uncertainty-aware relation-chain takeover.
```

## 3. Not Supported

V3.1 does not support:

```text
real-world takeover policy
human factors validation
real monitoring system robustness
general sensor-fusion safety
unrestricted relation discovery
deployment-ready engineering AI
```

## 4. Best One-Sentence Result

```text
V3.1 shows that the toy takeover diagnostic is not passed by generic overcaution or template-only audit; the agent must reject benign uncertainty, take over for concrete relation conflicts, and name the affected field and link.
```

