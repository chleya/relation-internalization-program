# V2.1 Self-Audit

## What this hardening improves

- Tests variable delay.
- Tests false temporal shortcut.
- Tests multi-link temporal edit.
- Requires temporal audit with time indexes.
- Reduces fixed-template false positives.

## Remaining weaknesses

- Candidate links are still predefined.
- Delay values are still from a small finite set.
- Toy world remains deterministic or near-deterministic.
- No real monitoring uncertainty model.
- No numerical slope mechanics.
- No real deployment claim.

## False positive risks

- Agent may learn candidate-link scoring heuristics.
- Agent may still exploit generator regularities.
- Audit strings may be formatted correctly without deep relation discovery.

## Next after V2.1

Do not jump to real deployment.
Potential V3: partial observability + missing sensors + delayed noisy observations + takeover threshold.
