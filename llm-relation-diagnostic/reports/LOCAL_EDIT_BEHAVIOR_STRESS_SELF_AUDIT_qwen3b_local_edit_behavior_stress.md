# LLM Relation Diagnostic Self-Audit

## What This Adds

- Defines a black-box JSON contract for LLM relation answers.
- Separates support-conditioned relation binding from global symbol mapping.
- Separates local edit behavior from edit acknowledgement.
- Separates relation-specific uncertainty from generic missingness templates.
- Keeps live LLM execution optional so the scoring layer can be tested without model noise.

## Current Passing Solvers

None.

## Remaining Weaknesses

- The task family is small and synthetic.
- Live model prompts may need hardening against invalid JSON.
- The scoring layer checks black-box behavior only; it does not inspect representations.
- A real LLM run should include multiple seeds and multiple local models before any claim is made.
