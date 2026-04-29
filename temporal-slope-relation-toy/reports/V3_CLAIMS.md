# V3 Claims

Date: 2026-04-28

## 1. Stage Claim

V3 supports this narrow claim:

```text
In the toy temporal slope relation environment, uncertainty-aware delayed relation-chain agents can reduce unsafe automation by triggering takeover under missing, noisy, delayed, or conflicting observations.
```

The stronger conceptual statement is:

```text
Temporal relation internalization is still insufficient unless it supports uncertainty-aware takeover.
```

## 2. Supported Claims

### 2.1 Relation-Chain Takeover Can Be Tested

V3 operationalizes takeover as an explicit action:

```text
monitor / drain / anchor / stop_work / takeover
```

This makes uncertainty a behavioral test, not just a report sentence.

### 2.2 Unsafe Automation Can Be Measured

V3 defines:

```text
unsafe_automation_rate =
cases requiring takeover where the agent still chooses an automatic action
```

This is stronger than ordinary action accuracy because it directly tests whether the system knows when not to act.

### 2.3 Structural Memory Remains A Strong Negative Control

Current mean result:

```text
structural_memory_temporal:
  noisy_action_success = 0.431
  takeover_precision = 0.726
  takeover_recall = 0.095
  unsafe_automation_rate = 0.905
  uncertain_relation_audit_score = 0.000
  gated_v3_score = 0.000
```

This supports:

```text
Temporal memory does not imply uncertainty-aware relation-chain takeover.
```

### 2.4 Delayed Relation-Chain Agents Pass V3 Gates

Current mean result:

```text
delayed_relation_chain:
  noisy_action_success = 0.759
  takeover_precision = 0.738
  takeover_recall = 0.971
  unsafe_automation_rate = 0.029
  uncertain_relation_audit_score = 1.000
  gated_v3_score = 0.886

learned_delayed_links:
  noisy_action_success = 0.759
  takeover_precision = 0.738
  takeover_recall = 0.971
  unsafe_automation_rate = 0.029
  uncertain_relation_audit_score = 1.000
  gated_v3_score = 0.886
```

This supports:

```text
The toy delayed relation-chain representation can support takeover behavior under the tested uncertainty regime.
```

## 3. Not Supported

V3 does not support:

```text
real geotechnical safety prediction
real monitoring system deployment
real-world takeover policy
unrestricted relation discovery
robustness to all sensor noise distributions
human factors validation
regulatory safety claims
```

## 4. Exact Boundary

Safe wording:

```text
V3 is a toy diagnostic for uncertainty-aware takeover in a delayed relation-chain setting.
```

Unsafe wording:

```text
V3 is an engineering safety model.
```

## 5. Best One-Sentence Result

```text
V3 shows that a temporal predictor can remain unsafe under uncertainty, while an auditable delayed relation-chain policy can trigger takeover and reduce unsafe automation in the toy setting.
```

