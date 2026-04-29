# Self-Audit

## What this experiment improves
- Moves from hand-written relation agents to neural relation emergence pressure.
- Tests whether relation structure can be extracted from learned hidden states.
- Tests editability and locality of extracted relations.
- Tests whether relation subspace is behaviorally causal.

## Remaining weaknesses
- Toy world remains simple.
- Edit pressure is still a designed training signal.
- Extracted table is based on canonical probing.
- Relation variables are known in the data generator.
- This is not unrestricted causal discovery.
- This is not proof of relation understanding in large neural models.

## False positive risks
- Model may learn canonical query patterns.
- Table extraction may overfit nuisance averaging.
- Edit-pressure architecture may impose too much relation-like structure.
- Probe intervention may remove correlated features, not pure relation factors.

## Required failure checks
1. Pure prediction passes all gates.
2. Edit-pressure model fails table edit.
3. Extracted table is correct only on canonical queries.
4. Relation subspace drop is small.
5. Nuisance subspace drop is large.
6. Edit locality is poor.
