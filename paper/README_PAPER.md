# README: Paper Draft

## What This Paper Draft Is

This directory contains a paper-style technical report draft for the Relation Internalization project.

Main title:

```text
When Prediction Is Not Relation Internalization:
Editable and Auditable Diagnostics for Relational Agents
```

Alternative title:

```text
Relation Internalization Beyond Prediction:
Toy Diagnostics for Editable, Auditable, and Cost-Aware Relational Agents
```

The draft presents the project as a diagnostic methodology. It is not written as a claim that relation reasoning, engineering safety, object permanence, or general causal discovery has been solved.

## Files

- `PAPER_DRAFT_V0.md`: full technical report draft.
- `CLAIMS_AND_LIMITATIONS.md`: supported and unsupported claims.
- `FIGURE_PLAN.md`: four-figure paper plan.
- `TABLE_FALSE_POSITIVE_LADDER.md`: false-positive elimination table.
- `README_PAPER.md`: this file.

## Project Reports and Results Summarized

The draft summarizes existing files only:

- `relation-internalization-test/reports/V1_RESEARCH_REPORT.md`
- `relation-internalization-test/results/summary.csv`
- `neural-relation-probe/reports/AUTO_REPORT.md`
- `neural-relation-probe/results/summary_modes.csv`
- `neural-relation-probe/results/extraction_summary.csv`
- `slope-relation-toy/reports/auto_report.md`
- `slope-relation-toy/results/summary.csv`
- `temporal-slope-relation-toy/reports/V2_TEMPORAL_REPORT.md`
- `temporal-slope-relation-toy/reports/V2_1_HARDENING_REPORT.md`
- `temporal-slope-relation-toy/reports/V2_1_FINAL_REPORT.md`
- `temporal-slope-relation-toy/results/temporal_summary.csv`
- `temporal-slope-relation-toy/results/hardening_summary.csv`
- `relation-agent-r1/reports/*.md`
- `relation-agent-r1/results/r1_summary.csv`
- `relation-agent-r1/results/r11_hardening_summary.csv`
- `relation-agent-r1/results/r12_discovery_summary.csv`
- `relation-agent-r1/results/r2_partial_observability_summary.csv`
- `relation-agent-r1/results/r2_1_summary.csv`
- `relation-agent-r1/results/r3_summary.csv`
- Existing figures under each project `figures/` directory.

No new experiments were added for this draft.

## What Not To Claim

Do not claim:

- real slope monitoring;
- real geotechnical safety capability;
- deployment-ready engineering AI;
- general causal discovery;
- object permanence;
- LLM replacement;
- proof that large neural systems naturally internalize relations;
- robustness to untested adversarial missingness;
- learned real-world sensor reliability;
- calibrated real-world inspection policy.

Prefer:

```text
passes diagnostic gates
```

over:

```text
understands relations
```

Prefer:

```text
relation structure is usable/editable/auditable
```

over:

```text
true causal understanding
```

## How To Update Numbers From CSVs

Use existing CSVs only. For each reported number:

1. Locate the stage result CSV.
2. Group by `agent`.
3. Compute the mean over seeds if the report uses means.
4. Copy only metrics already present in the CSV or generated report.
5. If a number is missing, write:

```text
[TODO: verify from result CSV/report]
```

Do not infer or invent a value.

Useful files:

- Static food-world: `relation-internalization-test/results/summary.csv`
- Neural probe: `neural-relation-probe/results/summary_modes.csv`
- Neural extraction: `neural-relation-probe/results/extraction_summary.csv`
- Slope toy: `slope-relation-toy/results/summary.csv`
- Temporal V2: `temporal-slope-relation-toy/results/temporal_summary.csv`
- Temporal V2.1: `temporal-slope-relation-toy/results/hardening_summary.csv`
- R1: `relation-agent-r1/results/r1_summary.csv`
- R1.1: `relation-agent-r1/results/r11_hardening_summary.csv`
- R1.2: `relation-agent-r1/results/r12_discovery_summary.csv`
- R2: `relation-agent-r1/results/r2_partial_observability_summary.csv`
- R2.1: `relation-agent-r1/results/r2_1_summary.csv`
- R3: `relation-agent-r1/results/r3_summary.csv`

## Next Step

The next step is a reviewer-style self-audit of `PAPER_DRAFT_V0.md`.

The audit should check:

- whether any claim exceeds the toy diagnostic evidence;
- whether each empirical number is traceable to a CSV/report;
- whether the false-positive ladder is central enough;
- whether limitations are strict enough;
- whether figure plans can be generated from existing figures/results without new experiments.
