# R1.1 Hardening Report

## 1. Motivation

R1.1 tests whether the R1 relation agent is more than a hand-fitted process learner.

## 2. Attacks

- Hidden nuisance/confounder candidates.
- Process-rule reversal after base training.
- Intervention cost tradeoff.
- Expanded candidate relation set with irrelevant links.
- Active-exploration ablation.

## 3. Results

| agent | n | hidden_confounder_rejection | reversal_adaptation | cost_tradeoff_success | candidate_expansion_precision | active_discovery_score | hardening_r11_gated_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 5 | 0.443 | 0.458 | 0.467 | 0.000 | 0.000 | 0.000 |
| shortcut | 5 | 0.920 | 0.903 | 0.000 | 0.000 | 0.000 | 0.000 |
| passive_memory | 5 | 0.777 | 0.707 | 0.667 | 0.000 | 0.000 | 0.000 |
| relation_no_explore | 5 | 0.777 | 0.707 | 0.667 | 1.000 | 0.000 | 0.000 |
| relation_agent | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 4. Interpretation

A passing agent must keep relation-guided action under nuisance shifts, adapt to a changed process rule, avoid unnecessary costly interventions, reject irrelevant candidate links, and actually perform active exploration.

## 5. Boundary

R1.1 remains a toy diagnostic. It is not unrestricted causal discovery, not an LLM replacement, and not real engineering competence.
