# Next Mainline Tasks

Date: 2026-04-29

## 1. Current Decision

Status update:

```text
V1 through V6.1 are implemented, tested, and frozen as diagnostic/shell stages.
R1 is implemented and tested.
R1.1 hardening is implemented and tested.
R1.2 discovery baseline is implemented and tested.
R2 partial-observability baseline is implemented and tested.
```

Current mainline:

```text
R2.1: Partial-Observability Hardening
```

Do not move back into generic review, governance, arbitration, or deployment
claims unless the task directly supports the agent's internal relation learning.

## 2. Why R1 Is The Mainline Now

The original research ambition was not to build better report templates. It was:

```text
find whether relation internalization can be implemented as a non-LLM route
toward usable intelligence.
```

V4-V6 are still useful, but they are shells around decisions. R1 returns to the
agent itself:

```text
intervene
observe transition
learn relation link
simulate counterfactual
choose action by relation graph
edit relation and change behavior
```

## 3. Implemented R1 Outputs

```text
relation-agent-r1\results\r1_summary.csv
relation-agent-r1\results\r1_records.csv
relation-agent-r1\results\r1_relations.json
relation-agent-r1\figures\r1_scores.png
relation-agent-r1\figures\r1_relation_recovery.png
relation-agent-r1\figures\r1_counterfactual_accuracy.png
relation-agent-r1\reports\R1_AGENT_REPORT.md
relation-agent-r1\reports\R1_SELF_AUDIT.md
relation-agent-r1\reports\R1_CLAIMS.md
relation-agent-r1\reports\R1_LIMITATIONS.md
```

## 4. Current R1 Result

```text
random: gated_r1_score = 0.000
shortcut: gated_r1_score = 0.000
passive_memory: gated_r1_score = 0.000
relation_agent: gated_r1_score = 0.972
```

R1.1:

```text
random: hardening_r11_gated_score = 0.000
shortcut: hardening_r11_gated_score = 0.000
passive_memory: hardening_r11_gated_score = 0.000
relation_no_explore: hardening_r11_gated_score = 0.000
relation_agent: hardening_r11_gated_score = 1.000
```

R1.2:

```text
random: discovery_r12_gated_score = 0.000
shortcut: discovery_r12_gated_score = 0.000
passive_memory: discovery_r12_gated_score = 0.000
relation_agent: discovery_r12_gated_score = 0.000
discovery_relation_agent: discovery_r12_gated_score = 1.000
```

R2:

```text
random: partial_r2_gated_score = 0.000
shortcut: partial_r2_gated_score = 0.000
passive_memory: partial_r2_gated_score = 0.000
discovery_relation_agent: partial_r2_gated_score = 0.000
uncertainty_discovery_agent: partial_r2_gated_score = 0.982
```

Important negative controls:

```text
shortcut has high in-distribution action_success but fails OOD shortcut reversal.
passive_memory has high action_success but fails relation recovery, counterfactual,
edit, and active-exploration gates.
```

## 5. Next Task: R2.1

R2 shows that a relation-discovery agent still fails under missing/noisy
observations unless it can inspect and audit uncertainty. The next weakness is:

```text
The uncertainty agent may pass by inspecting too often or by using a simple
missingness template rather than relation-specific uncertainty.
```

Required attacks:

```text
inspect-overuse penalty on fully observed low-risk states
irrelevant missing variable that should not trigger inspect
benign noise that does not affect the risk relation chain
audit-label-only negative control
fake uncertainty shortcut unrelated to relation-chain uncertainty
```

Required outputs:

```text
results/r21_uncertainty_hardening_summary.csv
figures/r21_uncertainty_hardening_score.png
reports/R2_1_UNCERTAINTY_HARDENING_REPORT.md
reports/R2_1_SELF_AUDIT.md
```

## 6. Boundary

Supported:

```text
toy active relation-learning diagnostic
non-LLM internal relation table
action by relation simulation
counterfactual and edit behavior
negative-control separation
```

Unsupported:

```text
LLM replacement
general artificial intelligence
real engineering agent
real safety prediction
unrestricted causal discovery
deployment-ready system
```
