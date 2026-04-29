# R1.2 Self-Audit

## What This Improves

- Removes direct dependence on the hand-written TRUE_LINK candidate table for the positive agent.
- Tests an unmarked nuisance setting rather than passing explicit nuisance labels into the agent.
- Tests discovery of a synthetic new process link.
- Uses adaptive low-coverage exploration instead of a fixed scripted schedule.

## Remaining Weaknesses

- The discovery agent still uses a process-variable schema.
- The environment is still small and deterministic.
- New-link discovery is synthetic and simple.
- The agent does not perform open-ended causal discovery.
- This is not an LLM replacement or real engineering intelligence.

## Current R1.2 Result

- discovery_relation_agent discovery_r12_gated_score: 1.000
