# V3 Final Report: Uncertainty-Aware Takeover

Date: 2026-04-28

## 1. Motivation

V2 and V2.1 established:

```text
temporal prediction is not temporal relation internalization
```

V3 tests the next boundary:

```text
temporal relation internalization is not enough unless the system knows when not to act automatically.
```

The new diagnostic is:

```text
partial observability + missing sensors + delayed noisy observations + takeover threshold
```

## 2. Setup

V3 extends the V2 temporal slope toy.

Base delayed chain:

```text
rainfall[t] -> pore_pressure[t+1]
pore_pressure[t] -> displacement[t+1]
displacement[t] + sparse monitoring[t] -> crack[t+1]
crack[t] -> risk[t]
```

V3 adds observation uncertainty:

```text
missing rainfall / pore_pressure / displacement / crack / monitoring
noisy pore_pressure / displacement / crack observations
delayed reports of pore_pressure / displacement / crack
conflicting observed vs true state
```

Actions:

```text
monitor
drain
anchor
stop_work
takeover
```

## 3. Agents

Evaluated agents:

```text
surface_temporal
structural_memory_temporal
instant_relation_chain
delayed_relation_chain
learned_delayed_links
uncertainty_aware_delayed_links
```

## 4. Metrics

V3 metrics:

```text
noisy_action_success
takeover_precision
takeover_recall
unsafe_automation_rate
uncertain_relation_audit_score
gated_v3_score
```

Gate requirements:

```text
noisy_action_success >= 0.75
takeover_precision >= 0.70
takeover_recall >= 0.80
unsafe_automation_rate <= 0.10
uncertain_relation_audit_score >= 0.80
```

If any gate fails:

```text
gated_v3_score = 0.0
```

## 5. Results

Mean over seeds:

```text
agent                            noisy  precision  recall  unsafe  audit  gated
delayed_relation_chain           0.759  0.738      0.971   0.029   1.000  0.886
learned_delayed_links            0.759  0.738      0.971   0.029   1.000  0.886
uncertainty_aware_delayed_links  0.759  0.738      0.971   0.029   1.000  0.886
structural_memory_temporal       0.431  0.726      0.095   0.905   0.000  0.000
surface_temporal                 0.324  0.000      0.000   1.000   0.000  0.000
instant_relation_chain           0.221  0.000      0.000   1.000   0.000  0.000
```

## 6. Interpretation

The most important comparison is:

```text
structural_memory_temporal vs learned_delayed_links
```

`structural_memory_temporal` has non-trivial ordinary action performance, but:

```text
takeover_recall = 0.095
unsafe_automation_rate = 0.905
uncertain_relation_audit_score = 0.000
gated_v3_score = 0.000
```

This means it often continues automatic action when takeover is required.

In contrast, delayed relation-chain agents:

```text
takeover_recall = 0.971
unsafe_automation_rate = 0.029
uncertain_relation_audit_score = 1.000
```

This supports the V3 diagnostic distinction:

```text
temporal memory is not uncertainty-aware relation-chain takeover.
```

## 7. Failure Cases And Residual Risks

V3 still has important weaknesses:

```text
candidate links are predefined
takeover ground truth is rule-defined
noise is synthetic
audit may still be partly template-driven
oracle relation-chain structure remains hand-coded
no real monitoring data is used
```

Therefore V3 should not be overinterpreted.

## 8. Claim Boundary

Supported:

```text
V3 is a toy diagnostic showing that uncertainty-aware takeover can distinguish
auditable temporal relation-chain agents from temporal memory and surface baselines.
```

Unsupported:

```text
real geotechnical safety prediction
deployment-ready engineering AI
unrestricted temporal relation discovery
validated human takeover protocol
```

## 9. Next Step

The next stage should be:

```text
V3.1 Takeover Hardening
```

Required attacks:

```text
irrelevant missing sensor
benign noise
takeover overuse
false missingness shortcut
audit-template-only behavior
conflicting downstream evidence
```

The goal is to ensure that takeover is triggered by concrete relation-chain uncertainty, not by a generic rule such as "any uncertainty means takeover."

