# Workshop Paper Outline V1

Target form:

```text
Workshop-style methodology paper / technical report
```

The paper should be written as a diagnostic benchmark/methodology contribution, not as a model paper.

## Title

When Prediction Is Not Relation Internalization:
Editable and Auditable Diagnostics for Relational Agents

## 1. Introduction

Goal:
Frame the problem as false-positive elimination.

Key points:

- Task success is too weak.
- Relation-internalization claims require usable structure.
- The paper offers staged toy diagnostics, not a general intelligence theory.
- Every stage includes a baseline that should pass if ordinary performance were enough.

Use `paper/INTRODUCTION_V1.md` as source.

## 2. Diagnostic Standard

Goal:
Define relation internalization operationally.

Include:

- transfer;
- counterfactual action;
- editability;
- auditability;
- temporal indexing;
- uncertainty recognition;
- cost-aware inspection.

Use `paper/METHODS_DIAGNOSTIC_CRITERIA_V1.md` as source.

## 3. False-Positive Ladder

Goal:
Make the paper's contribution concrete.

Include table of:

- prediction success;
- bottleneck compression;
- probe readability;
- structural memory;
- generic review text;
- temporal memory;
- fixed-delay template;
- relation discovery without uncertainty;
- blanket inspection;
- simple inspection heuristics;
- edit-signal responsiveness.

Use `paper/FALSE_POSITIVE_LADDER_TABLE.md`.

## 4. Experimental Groups

Goal:
Summarize groups without turning the paper into a project log.

Subsections:

1. Neural relation diagnostics.
2. Engineering-style relation chains.
3. Temporal relation diagnostics.
4. Active relation learning and discovery.
5. Partial observability and active inspection.

Keep each subsection to:

- what it tests;
- positive/focal model;
- false-positive controls;
- gate that matters.

## 5. Results

Goal:
Present the evidence ladder.

Main table:
Use `paper/MAIN_RESULT_TABLE.md`.

Narrative:
Use `paper/RESULTS_NARRATIVE_V1.md`.

Figure priority:

1. Evidence ladder.
2. Ordinary success vs gated score.
3. Edit/audit/counterfactual diagnostics.
4. Uncertainty and active inspection progression.

Use `paper/MAIN_FIGURE_EXECUTION_PLAN.md` and `paper/FIGURE_2_DATA_PLAN.md`.

## 6. Discussion

Goal:
Discuss false positives, not implementation details.

Use subsections:

- prediction false positive;
- compression false positive;
- probe readability false positive;
- editable behavior false positive;
- review text false positive;
- temporal memory false positive;
- discovery false positive;
- inspection false positive.

Use `paper/DISCUSSION_V1.md`.

## 7. Limitations

Must include:

- toy worlds;
- hand-specified variables;
- hand-designed positive controls in several stages;
- author-defined gates;
- counterfactual training is designed pressure;
- rule-based uncertainty and inspection policies;
- no real slope mechanics;
- no sensor calibration;
- no real engineering safety;
- no general causal discovery;
- no large-model behavior claim.

## 8. Future Work

Near-term:

- pre-registered or external gate variants;
- adversarial missingness;
- confidence calibration;
- learned uncertainty/value estimation;
- larger synthetic graphs.

Medium-term:

- neural policies with learned editable/auditable structures;
- learned sensor reliability;
- relation discovery without hand-declared variables.

Long-term:

- validated simulators before real engineering claims;
- benchmark families combining distribution shift, interventions, temporal delay, partial observability, and inspection cost.

## 9. Final Claim

Use this exact style:

```text
We present a staged toy diagnostic methodology showing that relation-internalization claims can be separated from prediction, probe readability, memory, surface shortcuts, review-like text, temporal prediction, blanket inspection, and edit-signal responsiveness. In these environments, agents pass diagnostic gates only when relation structure is usable for transfer, counterfactual action, edits, audits, uncertainty recognition, and cost-aware inspection.
```

## Do Not Include

- New experiments.
- New model claims.
- Claims that edit-pressure succeeds.
- Claims about real engineering systems.
- Claims about large language models.
- Claims that the gates are a full theory.
