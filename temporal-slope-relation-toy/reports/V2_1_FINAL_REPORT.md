# V2.1 Final Report

Date: 2026-04-28

## 1. Motivation

V2 proved that delayed temporal relation chains can be separated from same-step relation logic, surface shortcuts, and temporal state memory.

V2.1 hardens that result against reviewer concerns:

- fixed templates;
- single delay value;
- single-link edit;
- false temporal shortcut;
- audit strings without time indexes.

## 2. Added Hardening Tests

```text
variable_delay_success
false_delay_shortcut_rejection
multi_link_delay_edit_success
temporal_audit_consistency
anti_template_generalization
hardening_gated_score
```

## 3. Results

Mean over seeds 0-4:

```text
agent                       variable  shortcut  edit  audit  generalization  hardening
delayed_relation_chain      1.000     1.000     1.0   1.0    1.000           1.0
learned_delayed_links       1.000     1.000     1.0   1.0    1.000           1.0
structural_memory_temporal  0.985     0.980     0.0   0.0    0.975           0.0
instant_relation_chain      0.426     0.380     0.0   0.0    0.383           0.0
surface_temporal            0.296     0.291     0.0   0.0    0.291           0.0
```

## 4. Interpretation

`structural_memory_temporal` is the strongest negative control. It performs well on temporal prediction and generalization, but fails because it cannot edit delays and cannot audit temporal relations.

Therefore:

```text
temporal prediction is not temporal relation internalization.
```

## 5. Claim Boundary

Supported:

- V2.1 hardens the delayed relation-chain diagnostic against fixed-template shortcuts.
- Learned delayed links can pass the hardening gates in this toy setting.
- Structural temporal memory is not enough.

Unsupported:

- real geotechnical time-series modeling;
- unrestricted temporal relation discovery;
- deployment-ready engineering safety AI;
- safety-calibrated monitoring thresholds.

## 6. Freeze Decision

V2.1 is frozen as a stage result.

Further work should not add more V2.1 metrics. The next stage is V3:

```text
partial observability + missing sensors + delayed noisy observations + takeover threshold
```
