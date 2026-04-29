# R1 Relation Internalization Agent Report

## 1. Motivation

R1 returns to the core architecture: a non-LLM agent learns usable relations through interaction.

## 2. Results

| agent | n | action_success | ood_action_success | relation_recovery | counterfactual_accuracy | edit_success | active_exploration | gated_r1_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 5 | 0.507 | 0.472 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| shortcut | 5 | 0.918 | 0.215 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| passive_memory | 5 | 0.788 | 0.788 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| relation_agent | 5 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | 0.972 |

## 3. Interpretation

The relation agent must actively explore, recover internal links, act by relation simulation, answer counterfactuals, and respond to relation edits.

## 4. Boundary

R1 is a toy diagnostic. It is not a real engineering agent or general intelligence system.
