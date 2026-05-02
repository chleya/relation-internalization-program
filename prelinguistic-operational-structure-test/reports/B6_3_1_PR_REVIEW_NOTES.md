# B6.3.1 PR Review Notes

## PR Status

- PR: #3
- Base: main
- Head: b6-3-1-wrong-trace-mechanism-necessity
- Head commit: 635f79224bc449d6d05254e436cc78650560e4cd
- Status: OPEN
- Mergeable: MERGEABLE

## What This PR Adds

- B6.3.1 Wrong-Trace and Mechanism Necessity Refinement
- wrong_trace_no_public_state split
- targeted feedback/history necessity diagnostics
- delay5 credit buffer required diagnostics
- hidden indirect exploration/outcome-history diagnostics
- adversarial result review
- B-line evidence ledger
- B6.4 roadmap

## Validation

- pytest -q: 351 passed
- best_b6_3_1_refinement_score = 1.000
- mean_b6_3_1_refinement_score = 0.987
- b6_3_1_submit_ready_as_diagnostic = true
- feedback_required_drop = 0.275
- history_required_drop = 0.550
- credit_buffer_required_drop = 0.550
- git diff --check = clean

## Supported Evidence

- wrong_trace_no_public_state reduces state_only shortcut.
- targeted feedback/history splits show mechanism-specific drops.
- delay5 credit-buffer-required split shows drop under disabled buffer.
- hidden indirect discovery is separated from candidate-search fallback in synthetic diagnostics.
- B-line evidence ledger now records the chain from B5 through B6.3.1.

## Claim Boundary

This PR supports only toy PLOS diagnostic evidence.

It does not support:

- solved robust trace repair
- general feedback/history necessity
- general delayed credit solved
- real-world causal discovery
- real-world risk intelligence
- robotics capability
- safety certification
- construction-site autonomy
- deployable engineering control

## Remaining Risks

- hidden_indirect_discovery_score = 1.000 depends on synthetic exploration/outcome-history cues.
- high aggregate score must not be treated as general structural necessity proof.
- B6.4 is needed to test transfer and anti-overfit generalization.

## Recommendation

Keep PR #3 open for review or merge only with caveats preserved. Do not proceed to B7. The next experimental stage should be B6.4 Transfer and Anti-Overfit Generalization on a new branch.
