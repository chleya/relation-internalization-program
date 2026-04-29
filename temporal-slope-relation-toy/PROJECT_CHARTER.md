# Project Charter: Temporal Slope Relation Toy

Date: 2026-04-28

## Purpose

V2 tests delayed process relation internalization.

The project asks:

```text
Can a system use a delayed physical relation chain rather than same-step rules, temporal memory, or surface warnings?
```

## Core Delayed Chain

```text
rainfall[t] -> pore_pressure[t+1]
pore_pressure[t] -> displacement[t+1]
displacement[t] + monitoring[t] -> crack/risk[t+1]
```

## Non-Goals

- Do not use real slope monitoring data.
- Do not claim real geotechnical time-series modeling.
- Do not attempt unrestricted relation discovery.
- Do not add engineering review text metrics.

## Agents

```text
surface_temporal
structural_memory_temporal
instant_relation_chain
delayed_relation_chain
learned_delayed_links
```

## Gates

```text
temporal_ood_success >= 0.8
delayed_counterfactual_accuracy >= 0.8
delay_edit_success >= 0.9
surface_shortcut_rejection >= 0.8
temporal_audit_score >= 0.9
```

## Current Claim

Supported:

```text
The toy distinguishes delayed relation-chain use from surface temporal shortcuts, same-step relation logic, and temporal state memory.
```

Not supported:

```text
real engineering safety decisions
real geotechnical process modeling
unrestricted temporal relation discovery
```
