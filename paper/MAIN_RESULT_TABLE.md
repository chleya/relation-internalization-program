# Main Result Table

| Stage | Positive / focal result | Gated score or key metric | False-positive controls | Conservative interpretation |
| --- | --- | ---: | --- | --- |
| Neural probe | base neural relation model | gated score about `0.819` | shortcut neural model `0.000` | Probe readability must be paired with behavioral specificity. |
| Neural-to-table extraction | base extracted table | gated extraction `1.000` | shortcut extracted table `0.000` | Extracted tables must transfer and reject shortcuts. |
| Neural counterfactual training | `counterfactual_training` | `~0.981` | `pure_prediction` `0.000`, `prediction_bottleneck` `0.000` | Counterfactual pressure is strongest current non-handwritten neural positive condition. |
| Neural edit pressure | `edit_pressure_training` | `~0.200` | edit-signal responsiveness without binding | Editability is mixed and can be a false positive. |
| Slope toy | `relation_chain` / `learned_links` | about `0.98` | `structural_memory` `0.000`, `generic_review` `0.000` | Relation chains beat memory and review-like text under gates. |
| Temporal V2/V2.1 | `delayed_relation_chain`, `learned_delayed_links` | `1.000` | `structural_memory_temporal` `0.000`, `instant_relation_chain` `0.000` | Temporal relation structure must be editable and auditable. |
| R1 | `relation_agent` | `0.972` | random/shortcut/passive `0.000` | Active relation behavior requires more than action success. |
| R1.1 | hardened `relation_agent` | `1.000` | no-explore relation table `0.000` | Exploration and hardening controls matter. |
| R1.2 | `discovery_relation_agent` | `1.000` | hand-supplied relation dependence | Candidate discovery must come from transitions. |
| R2 | `uncertainty_discovery_agent` | `0.982` | discovery-only: success `0.967`, gated `0.000` | Discovery without uncertainty can automate unsafely. |
| R2.1 | relation-specific uncertainty agent | `0.933` | `missing_always_inspect` `0.000` | Blanket inspection fails precision and cost. |
| R3 | active inspection agent | `0.933` | random/first/missing/risk-first `0.000` | Inspection must target informative fields under budget. |

These numbers are anchors from existing reports and result CSVs. They should not be generalized beyond the toy diagnostic settings.
