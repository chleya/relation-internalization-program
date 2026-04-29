# False-Positive Ladder Table

| False positive | Example baseline / result | Looks strong because | Rejected by |
| --- | --- | --- | --- |
| prediction success | `pure_prediction` | high train/OOD accuracy | shortcut rejection and relation-subspace gates |
| bottleneck compression | `prediction_bottleneck` | compact representation and strong behavior metrics | nuisance subspace remains entangled |
| probe readability | shortcut neural probe | relation labels are readable | OOD/spurious/subspace behavior fails |
| extracted but wrong table | shortcut extracted table | canonical extraction can look structured | OOD/spurious extraction gates |
| structural memory | slope `structural_memory` | OOD/spurious success can be high | edit/audit/review gates |
| generic review text | `generic_review` | plausible review language | relation behavior and consistency gates |
| temporal memory | `structural_memory_temporal` | high temporal prediction | delay edit and temporal audit gates |
| fixed-delay template | fixed delay / instant relation controls | works in narrow timing cases | variable-delay and anti-template hardening |
| relation discovery without uncertainty | R2 `discovery_relation_agent` | partial observation success `0.967` | unsafe automation and uncertainty audit gates |
| blanket inspection | `missing_always_inspect` | appears conservative | inspection precision, unnecessary inspection, cost gates |
| simple inspection heuristic | first-missing/random/risk-first | sometimes selects useful fields | target accuracy, information gain, budgeted safety gates |
| edit-signal responsiveness | V1.2 `edit_pressure_training` | `edit_state_swap_success = 1.000` | support shuffle, binding, and causal subspace diagnostics |

The added neural-stage lesson is:

```text
edit-signal responsiveness != support-conditioned relation internalization
```
