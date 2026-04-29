# Figure 2 Data Plan: Ordinary Success vs Gated Relation Score

## Purpose

Figure 2 should make the paper's central claim visually immediate:

```text
ordinary success != relation internalization
```

The figure should use existing results only. It should not introduce new runs, softer gates, or tuned baselines.

## Recommended Design

Use a paired bar or scatter plot:

- x-axis: ordinary success-like metric
- y-axis: gated relation score
- each point/bar pair is one agent or baseline
- annotate false positives where ordinary score is high but gated score is zero or low

The strongest layout is paired bars because the contrast is easier to read:

```text
agent/baseline | ordinary metric | gated score
```

## Core Data Points

| Label | Ordinary metric | Ordinary value | Gated score | Source | Message |
| --- | --- | ---: | ---: | --- | --- |
| R2 discovery-only | partial observation success | `0.967` | `0.000` | R2 reports/results | Relation discovery without uncertainty can automate unsafely. |
| Temporal structural memory | temporal prediction / temporal success | high; exact value to verify from CSV/report | `0.000` | V2/V2.1 reports | Temporal prediction is not editable delayed relation structure. |
| R2.1 missing-always | critical-missing inspection tendency / recall | high; exact value to verify from CSV/report | `0.000` | R2.1 reports/results | Blanket inspection is not relation-specific uncertainty. |
| Neural edit-pressure | table/edit/locality metrics and edit swap | table/edit/locality strong; `edit_state_swap_success = 1.000` | `~0.200` | neural stage summary, V1.2 diagnostics | Edit responsiveness is not support-conditioned relation internalization. |
| Neural counterfactual | neural diagnostic metrics | OOD/shortcut/reversal/table/edit/locality strong | `~0.981` | neural stage summary | Current strongest non-handwritten neural positive condition. |
| Slope structural memory | OOD/spurious performance | high; exact value to verify from CSV/report | `0.000` | slope toy reports | Structural memory is not editable/auditable relation structure. |

## Optional Positive Reference Points

Use these as anchors if the figure needs context:

| Label | Ordinary metric | Gated score | Message |
| --- | ---: | ---: | --- |
| R2 uncertainty agent | high partial-observation performance | `0.982` | Discovery paired with uncertainty passes. |
| R2.1 relation-specific uncertainty | cost-adjusted / gated performance | `0.933` | Relation-specific uncertainty beats blanket inspection. |
| R3 active inspection | budgeted active inspection metrics | `0.933` | Inspection selection must be target-aware and cost-aware. |
| Temporal delayed links | temporal performance | `1.000` | Delayed relation structure passes hardening. |

## Caption Draft

Ordinary task-like success does not imply relation internalization. Several baselines achieve high prediction, temporal, partial-observation, inspection, or edit-response metrics while receiving zero or low gated scores because they fail structural diagnostics such as auditability, uncertainty recognition, support-conditioned binding, or cost-aware inspection.

## Notes for Missing Exact Values

Some ordinary metric values should be verified from existing CSVs/reports before plotting:

- `structural_memory_temporal` ordinary temporal metric;
- R2.1 `missing_always_inspect` inspection recall or critical-missing inspection tendency;
- slope `structural_memory` OOD/spurious metric.

If a value is not directly available, do not fabricate it. Use the gated score table or mark the ordinary metric as "reported high; verify exact value".

## Do Not Include

- Do not average across incompatible ordinary metrics without labeling them.
- Do not imply that the x-axis is one universal accuracy measure.
- Do not hide `edit_pressure_training`; its mixed result is important.
- Do not show edit-pressure as a positive result.
