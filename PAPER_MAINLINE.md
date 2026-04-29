# Paper Mainline: Relation Internalization as Editable, Auditable, and Cost-Aware Structure

Date: 2026-04-29

## 1. Proposed Title

**Relation Internalization Beyond Prediction: Toy Diagnostics for Editable, Auditable, and Cost-Aware Relational Agents**

Alternative shorter title:

**When Prediction Is Not Relation Internalization**

## 2. Abstract Draft

We study a narrow question: when should an agent be credited with internalizing a relation rather than merely predicting outcomes or exploiting surface regularities? We build a sequence of toy diagnostics that progressively rule out common false positives. The early neural probe shows that readable relation information is not sufficient unless the representation is behaviorally relevant and extractable into an editable relation table. The slope toy shows that explicit relation chains can support counterfactual action, edits, audits, and shortcut rejection, while generic review or structural memory alone fails the gated score. Temporal V2/V2.1 extends the criterion to delayed relations and rules out same-step templates, fixed-delay shortcuts, and temporal memory without editable delay structure. R1 then resets the line to a minimal non-LLM active agent that learns relations through intervention, uses them for action and counterfactuals, and adapts to relation edits. R2/R2.1 show that relation discovery alone is not enough under partial observability: the agent must know when relation evidence is unverifiable and must avoid blanket inspection. R3 adds active inspection selection under cost, requiring the agent to choose informative fields, update sequentially, and respect a limited inspection budget.

Across these stages, the supported claim is limited but precise: in small diagnostic worlds, relation internalization can be operationalized as usable, transferable, editable, auditable, and uncertainty-aware structure. The work does not claim real slope monitoring, general causal discovery, object permanence, LLM replacement, or deployment-ready engineering AI.

## 3. Core Thesis

The paper should not claim that the system solves engineering intelligence. The paper should claim that **relation internalization needs stronger evidence than task accuracy**.

The proposed diagnostic standard is:

```text
A relation is internalized only if it supports transfer, counterfactual use,
intervention/edit, audit, uncertainty handling, and cost-aware information
gathering under negative controls.
```

This lets the paper make a clean contribution: not a new large model, but a staged test suite and minimal agents that expose false positives.

## 4. Evidence Chain

| Stage | Positive Result | Alternative Explanation Ruled Out |
| --- | --- | --- |
| Neural probe | Base neural model has readable, behaviorally relevant relation subspace; extracted table works under OOD/spurious attack. | Probe readability alone; hidden shortcut features; editable but wrong extracted table. |
| Slope toy | Relation-chain agents pass OOD, spurious attack, counterfactual, edit, audit, review consistency gates. | Majority behavior; surface feature policy; structural memory without editable/auditable relations; generic review text. |
| Temporal V2 | Delayed relation chains pass temporal OOD, delayed counterfactual, delay edit, temporal audit. | Same-step relation logic; surface temporal shortcut; temporal state memory without delay edits/audits. |
| Temporal V2.1 | Variable delay, false shortcut rejection, multi-link delay edit, audit consistency, anti-template generalization pass for delayed-link agents. | Fixed delay template; single hard-coded delay; single-link edit trick; audit string without temporal structure. |
| R1 | Active relation agent learns usable links through interaction and passes action, OOD, relation recovery, counterfactual, edit, active exploration gates. | Random policy; shortcut policy; passive memory; prediction without internal links. |
| R1.1 | Relation agent survives hidden confounder, rule reversal, cost tradeoff, candidate expansion, active discovery controls. | Hand-fitted process learner; nuisance feature dependence; irrelevant candidate acceptance; no-exploration relation table. |
| R1.2 | Discovery agent learns relation candidates from observed transitions and passes nuisance rejection, new-link discovery, adaptive exploration, precision, action gates. | Predefined TRUE_LINK table; hand-supplied candidate graph; fixed exploration schedule. |
| R2 | Uncertainty discovery agent passes partial observability gates; discovery-only agent gets zero gated score despite high action success. | Relation discovery is sufficient; high action success under partial input proves safety; memory/prediction is enough. |
| R2.1 | Relation-specific uncertainty agent beats missing-always-inspect under cost and precision; noncritical missingness does not force inspection. | Blanket “if unknown then inspect” template; full-state oracle inspection; vague uncertainty audit. |
| R3 | Active inspection agent selects informative fields, updates sequentially, avoids unsafe automation and overinspection, and beats first/random inspection baselines. | Inspect-first heuristic; random field choice; risk-first without budget discipline; safe behavior from unlimited inspection. |

## 5. Key Empirical Anchors

Use only existing results.

Neural probe:

- Base mode: OOD `1.000`, spurious attack `1.000`, probe selectivity about `0.793`, relation-subspace drop about `0.483`, gated neural relation score about `0.819`.
- Shortcut mode: readable relation probe remains high, but OOD/spurious robustness collapses and gated score is `0.000`.
- Extracted table: base gated extraction `1.000`; shortcut gated extraction `0.000`.

Slope toy:

- `relation_chain` / `learned_links`: gated slope score about `0.98`.
- `structural_memory`: OOD and spurious performance are high, but edit/audit/review gates fail, so gated score is `0.000`.
- `generic_review`: review-like output does not recover relation behavior; gated score `0.000`.

Temporal V2/V2.1:

- `delayed_relation_chain` and `learned_delayed_links`: gated temporal and hardening scores `1.000`.
- `structural_memory_temporal`: strong temporal prediction but no delay edit or temporal audit, hardening score `0.000`.
- `instant_relation_chain`: fails delayed structure, hardening score `0.000`.

R1-R3:

- R1 `relation_agent`: gated R1 score `0.972`; shortcut/passive/random score `0.000`.
- R1.1 `relation_agent`: hardening score `1.000`; relation without exploration score `0.000`.
- R1.2 `discovery_relation_agent`: discovery score `1.000`; original relation agent without discovery score `0.000`.
- R2 `discovery_relation_agent`: partial observation success `0.967` but gated score `0.000`; `uncertainty_discovery_agent` gated score `0.982`.
- R2.1 `relation_specific_uncertainty_agent`: gated score `0.933`; `missing_always_inspect` score `0.000`.
- R3 `active_inspection_agent`: gated score `0.933`; random/first/missing/risk-first inspection baselines score `0.000`.

## 6. Paper Contributions

1. **A stricter operational definition of relation internalization.** The paper defines relation internalization through transfer, counterfactual action, editability, auditability, uncertainty handling, and cost-aware inspection rather than task accuracy alone.

2. **A staged negative-control methodology.** Each stage introduces a false-positive baseline that can look strong on ordinary success metrics but fails a gated diagnostic.

3. **A neural-to-symbolic bridge result.** The neural probe shows when relation information can be read from a hidden state and extracted into an editable table, while also showing that shortcut models can be probe-readable but not relation-internalized.

4. **A temporal relation diagnostic.** V2/V2.1 show that delayed relation internalization is separable from same-step logic and temporal memory.

5. **An active partial-observability line.** R2/R2.1/R3 show that relation discovery must be extended with uncertainty, relation-specific inspection, and budgeted active information selection.

## 7. Experimental Structure

### Experiment 1: Neural Readability Is Not Enough

Purpose:
Test whether a small neural classifier encodes relation information and whether that information is causally used.

Key comparison:

- Base relation-trained model.
- Shortcut-trained model.

What it rules out:

- A high probe score alone does not prove relation internalization.
- An editable extracted table does not prove correctness unless it transfers and aligns with true relations.

Main result:
Base passes; shortcut fails gated neural and extraction scores.

### Experiment 2: Explicit Relation Chains Beat Review Shells and Surface Policies

Purpose:
Move from neural representation to a toy slope relation chain with counterfactuals, edits, audits, and review consistency.

Key comparison:

- Surface policy.
- Majority policy.
- Structural memory.
- Generic review.
- Learned links / relation chain.

What it rules out:

- Good prediction from structural memory is not enough.
- Review-like text is not enough.
- Surface robustness under easy cases is not relation use.

Main result:
Relation-chain agents pass; structural memory and generic review fail gates.

### Experiment 3: Temporal Relations Require Editable Delay Structure

Purpose:
Test delayed relation chains.

Key comparison:

- Surface temporal.
- Instant relation chain.
- Structural temporal memory.
- Delayed relation chain.
- Learned delayed links.

What it rules out:

- Same-step relation logic.
- Fixed delay templates.
- Temporal memory without editable/auditable delay relations.

Main result:
Delayed-link agents pass V2 and V2.1; structural temporal memory fails because it cannot edit or audit delays.

### Experiment 4: Active Relation Learning

Purpose:
Return to a minimal non-LLM agent that learns usable relations through interaction.

Key comparison:

- Random.
- Shortcut.
- Passive memory.
- Relation agent.
- Relation no-explore.
- Discovery relation agent.

What it rules out:

- Passive success is not relation internalization.
- Predefined candidate links are not enough.
- No-exploration tables are not enough.
- Discovery must come from transition evidence.

Main result:
R1, R1.1, and R1.2 isolate active learning, hardening, and relation discovery.

### Experiment 5: Partial Observability Requires Uncertainty

Purpose:
Test whether discovered relations can be used safely when critical observations are missing, noisy, or conflicting.

Key comparison:

- Discovery relation agent.
- Uncertainty discovery agent.

What it rules out:

- High action success under partial observation is not enough.
- Relation discovery alone is not enough.
- Acting under unverifiable relation chains is unsafe automation.

Main result:
Discovery-only agent gets high partial success but zero gated score; uncertainty-aware agent passes.

### Experiment 6: Relation-Specific Inspection and Active Field Selection

Purpose:
Test whether inspection is targeted and budget-aware.

Key comparison:

- Missing-always-inspect.
- First-missing-inspect.
- Random-inspect-field.
- Risk-first-inspect.
- Relation-specific uncertainty agent.
- Active inspection agent.

What it rules out:

- Blanket inspection.
- First missing field heuristic.
- Risk-first heuristic without budget discipline.
- Full-state oracle inspection.

Main result:
R2.1 separates relation-specific uncertainty from blanket inspection; R3 adds active field selection under cost.

## 8. Main Figure Design

### Figure 1: Evidence Ladder

One horizontal pipeline:

```text
Neural probe -> Slope relation chain -> Temporal delayed chain -> R1 active learner -> R2 uncertainty -> R3 active inspection
```

Under each node, show the false positive eliminated:

```text
probe readability
surface/review shell
same-step memory
passive/shortcut success
discovery without uncertainty
blanket inspection
```

Purpose:
This is the paper’s conceptual figure.

### Figure 2: Gated Score Summary

Bar chart with one positive method and major negative controls per stage:

- Neural base vs shortcut.
- Slope relation chain vs structural memory/generic review.
- Temporal learned delayed links vs structural temporal memory.
- R1 relation/discovery agent vs shortcut/passive.
- R2 uncertainty agent vs discovery-only.
- R3 active inspection vs first/random/missing baselines.

Purpose:
Show that ordinary-looking baselines are zeroed by hard gates.

### Figure 3: Prediction Accuracy vs Relation Internalization

Scatter or paired bars:

- x-axis: ordinary success metric.
- y-axis: gated relation score.

Important examples:

- `discovery_relation_agent` in R2: high partial observation success, gated `0.000`.
- `structural_memory_temporal`: high temporal success, hardening `0.000`.
- `shortcut` neural model: readable relation probe, gated `0.000`.

Purpose:
This figure makes the main argument visually: prediction/readability is not enough.

### Figure 4: Edit and Audit Requirements

Grouped bars for:

- counterfactual accuracy;
- edit success;
- audit score;
- relation recovery/discovery.

Use slope toy, temporal V2/V2.1, and R1.

Purpose:
Show why the claim is about internal structure rather than external behavior.

### Figure 5: Partial Observability and Inspection

Panel A:
R2 inspection recall, unsafe automation rate, uncertainty audit score.

Panel B:
R2.1 inspection precision vs unnecessary inspection.

Panel C:
R3 inspection target accuracy, information gain efficiency, budgeted safe action rate.

Purpose:
Show the progression from uncertainty recognition to targeted, cost-aware information gathering.

## 9. Claim Boundary

Supported:

- In toy diagnostic environments, relation internalization can be distinguished from prediction, memory, surface shortcuts, and review-like explanation.
- Explicit relation structures can support counterfactual action, relation edits, and audits.
- Delayed relation chains require editable temporal structure, not just temporal prediction.
- Under partial observability, relation discovery must be paired with uncertainty handling.
- Under cost and limited budget, inspection must be targeted by expected relation value rather than blanket conservatism.

Not supported:

- Real slope monitoring.
- Real geotechnical engineering safety.
- Deployment-ready engineering AI.
- General causal discovery.
- Object permanence.
- LLM replacement.
- Claims that large neural systems naturally internalize relations.
- Robustness to adversarial missingness beyond the tested toy cases.
- Learned sensor reliability calibration.
- Real-world calibrated inspection policy.

Preferred wording:

```text
The experiments support a diagnostic standard for relation internalization,
not a deployed relation-intelligent system.
```

Avoid:

```text
solves relation reasoning
solves object permanence
passes SVT
real slope safety model
general causal understanding
```

## 10. Discussion Mainline

The discussion should be organized around false positives.

1. **Probe false positive:** A relation can be linearly readable but not behaviorally specific.
2. **Prediction false positive:** An agent can predict well without editability or auditability.
3. **Temporal memory false positive:** An agent can remember temporal patterns without internal delayed links.
4. **Discovery false positive:** An agent can discover relations but still act unsafely when the current relation chain is unverifiable.
5. **Inspection false positive:** An agent can look safe by inspecting everything, but fail under cost and limited budget.

The paper’s strongest sentence:

```text
Every stage was designed around a baseline that should pass if ordinary
performance were enough, but fails once relation internalization is required.
```

## 11. Recommended Paper Organization

1. Introduction
   - Why task accuracy is too weak.
   - Define relation internalization operationally.
   - Preview staged diagnostics and negative controls.

2. Diagnostic Criteria
   - Transfer.
   - Counterfactual use.
   - Editability.
   - Auditability.
   - Temporal indexing.
   - Uncertainty under partial observability.
   - Cost-aware inspection.

3. Experimental Worlds and Agents
   - Neural probe world.
   - Slope relation toy.
   - Temporal slope toy.
   - Active relation agent R1-R3.
   - Baselines and gated scores.

4. Results
   - Neural readability/extraction.
   - Slope relation chains.
   - Temporal delayed links.
   - Active relation learning.
   - Partial observability.
   - Active inspection.

5. What Each Stage Rules Out
   - This should be a central section, not appendix material.

6. Limitations
   - Toy worlds.
   - Hand-specified variables.
   - Rule-based uncertainty/inspection policies.
   - No real sensor noise calibration.
   - No deployment claims.

7. Future Work

## 12. Future Work

Near-term:

- Replace rule-based inspection value with learned uncertainty/value estimation while preserving the same gates.
- Add adversarial missingness and correlated sensor failures to R3-style active inspection.
- Test whether learned relation discovery and inspection selection remain stable under larger variable graphs.
- Add calibration curves for confidence, not only gated pass/fail metrics.
- Connect neural-to-table extraction with the R1-R3 active inspection line.

Medium-term:

- Study whether neural policies can learn the same editable/auditable relation structures without hand-written relation tables.
- Introduce noisy, delayed, and partially reliable inspection channels with learned sensor reliability.
- Evaluate multi-agent audit of relation uncertainty without turning the system into a governance shell.

Long-term:

- Move from toy process worlds to validated simulators.
- Define what evidence would be required before any engineering safety claim.
- Build benchmark families where relation internalization must survive distribution shift, intervention, temporal delay, partial observability, and inspection cost together.

## 13. Final Paper Claim

Use this as the narrow final claim:

```text
We present a staged toy diagnostic framework showing that relation
internalization can be separated from prediction, probe readability, memory,
surface shortcuts, and blanket inspection. In these environments, agents pass
only when relation structure is usable for transfer, counterfactual action,
edits, audits, uncertainty recognition, and cost-aware inspection.
```

Do not claim more than this.
