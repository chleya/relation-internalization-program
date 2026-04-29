# Main Figure Execution Plan

This plan freezes the main paper to four figures. No new experiments should be run for these figures. Use existing CSVs, reports, and already generated summaries.

## Figure 1: Evidence Ladder

Purpose:
Show the paper as a staged diagnostic methodology rather than a project log.

Design:
A horizontal ladder with six nodes:

```text
Static relation diagnostics
-> Neural relation diagnostics
-> Engineering-style relation chains
-> Temporal relation diagnostics
-> Partial observability
-> Active inspection
```

Under each node, show the false positive eliminated:

| Node | False positive eliminated |
| --- | --- |
| Static relation diagnostics | memory, fitting, shortcut policies, predefined-link dependence |
| Neural relation diagnostics | prediction, bottleneck compression, probe readability, edit-signal responsiveness |
| Engineering-style relation chains | structural memory, surface labels, generic review text |
| Temporal relation diagnostics | temporal memory, same-step logic, fixed-delay templates |
| Partial observability | relation discovery without uncertainty |
| Active inspection | blanket inspection, first/random/risk-first heuristics |

Recommended source files:

- `paper/MAIN_PAPER_DRAFT_V0.md`
- `paper/FALSE_POSITIVE_LADDER_TABLE.md`
- `paper/UPDATED_FALSE_POSITIVE_LADDER_ADDENDUM.md`

## Figure 2: Ordinary Success vs Gated Relation Score

Purpose:
Make the central claim visually obvious: ordinary performance can be high while gated relation score is zero.

Design:
Scatter or paired-bar plot:

- x-axis: ordinary metric
- y-axis: gated score
- annotate false-positive points

Required points:

| Example | Ordinary metric | Gated score |
| --- | ---: | ---: |
| `discovery_relation_agent` in R2 | partial observation success `0.967` | `0.000` |
| `structural_memory_temporal` | high temporal prediction | `0.000` |
| `missing_always_inspect` | high critical inspection tendency | `0.000` |
| `edit_pressure_training` | table/edit/locality strong; edit swap `1.000` | `~0.200` |
| `counterfactual_training` | neural diagnostic metrics strong | `~0.981` |

Recommended source files:

- R2 report/result CSVs
- temporal V2/V2.1 reports
- `neural-relation-internalization-edit-pressure/results/summary.csv`
- `neural-relation-internalization-edit-pressure/results/interaction_diagnostics_summary.csv`

Caption emphasis:

```text
High ordinary performance is not credited as relation internalization unless structural gates pass.
```

## Figure 3: Edit / Audit / Counterfactual Diagnostics

Purpose:
Show that relation-internalization claims require usable structure, not only output success.

Design:
Grouped bars across representative agents:

- counterfactual accuracy
- edit success
- audit or relation alignment score
- gated score

Suggested groups:

| Group | Positive | Negative control |
| --- | --- | --- |
| Static/R1 | `relation_agent` | shortcut/passive |
| Slope toy | `relation_chain` / `learned_links` | `structural_memory`, `generic_review` |
| Temporal | `learned_delayed_links` | `structural_memory_temporal`, `instant_relation_chain` |
| Neural | `counterfactual_training` | `edit_pressure_training` as mixed editability false positive |

Caption emphasis:

```text
Editability and auditability are necessary diagnostics, but neural edit responsiveness alone is not sufficient.
```

## Figure 4: Uncertainty and Active Inspection Progression

Purpose:
Show the R2 -> R2.1 -> R3 progression from uncertainty recognition to targeted, cost-aware inspection.

Design:
Three-panel figure:

Panel A: R2

- `discovery_relation_agent`: partial observation success high, gated `0.000`
- `uncertainty_discovery_agent`: gated `0.982`

Panel B: R2.1

- `missing_always_inspect`: gated `0.000`
- `relation_specific_uncertainty_agent`: gated `0.933`
- show inspection precision / unnecessary inspection where available

Panel C: R3

- random / first-missing / missing-always / risk-first baselines: gated `0.000`
- `active_inspection_agent`: gated `0.933`
- show inspection target accuracy or budgeted safe action rate where available

Caption emphasis:

```text
Relation discovery must be paired with uncertainty recognition and budgeted inspection selection.
```

## Appendix Figures

Move all other generated plots to appendix unless space allows:

- neural probe details
- V1.1 edit-pressure failure localization
- V1.2 interaction diagnostics
- per-stage gate breakdowns
- ablation matrices

## Do Not Do

- Do not run new experiments.
- Do not tune models for prettier figures.
- Do not replace gated scores with softer metrics.
- Do not describe edit-pressure as a stable success.
- Do not imply real engineering safety.
