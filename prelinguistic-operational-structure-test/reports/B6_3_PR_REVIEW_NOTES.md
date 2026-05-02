# B6.3 PR Review Notes

## PR Status

- PR: #2
- base: main
- head: b6-3-structural-necessity
- commit: 11e0c57
- status: open
- merged: no
- review decision: acceptable as a diagnostic branch only

## Validation

Commands rerun locally:

```bash
python -m src.run_b6_3_structural_necessity --config configs/b6_3_structural_necessity.yaml --seed 0
python -m src.run_b6_3_result_review
pytest -q
python -m src.visualize_b6_3 --summary results/b6_3_structural_necessity_summary.csv
git diff --check
```

Observed results:

- best_b6_3_structural_necessity_score = 0.997
- mean_b6_3_structural_necessity_score = 0.620
- b6_3_submit_ready_as_diagnostic = true
- b6_3_trace_necessity_drop = 0.124
- b6_3_candidate_search_drop = 0.127
- pytest: 342 passed
- git diff --check: clean
- forbidden_reference_count_max = 0
- invalid_metric_count_total = 0
- poisoned_ground_truth_invariance_all_pass = true

## What This PR Adds

- structural ablation diagnostics
- trace repair / confidence downgrade diagnostics
- candidate search necessity checks
- delayed credit buffer ablation
- history, feedback, and risk cue ablations
- baseline comparisons against state_only, mask_only, trace_only, conservative, random, always_abstain, and oracle
- result review artifacts
- reports and visualization

## Evidence Supported

- Trace ablations cause a moderate aggregate drop: remove_trace = 0.124, shuffle_trace = 0.124, corrupt_trace = 0.124.
- Candidate search ablation causes a moderate aggregate drop: disable_candidate_search = 0.127.
- Risk cue removal causes the largest observed aggregate drop: remove_risk_cue = 0.179.
- wrong_trace_state_ambiguous reduces the state_only shortcut: b63_policy = 1.000, state_only = 0.753, mask_only = 0.690.
- B6.3 improves diagnostic clarity around which mechanisms do and do not show necessity evidence.

## Evidence Not Supported

- Robust trace repair is not solved.
- Private trace necessity is not proven.
- Feedback update necessity is not proven.
- History necessity is not proven.
- Delayed credit buffer necessity is weak in aggregate.
- Hidden indirect causal path discovery is not proven.
- hide_public_state_cue aggregate drop = 0.000 does not prove that public-state dependence has been eliminated.
- No real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or engineering deployment is supported.

## Blocking Weaknesses

1. wrong_trace remains explainable by state_only: b63_policy = 0.753, state_only = 1.000, mask_only = 0.959.
2. hide_public_state_cue does not collapse aggregate performance, because history and feedback support remain available.
3. feedback/history drops are weak: freeze_feedback_update = 0.019, remove_history = 0.025.
4. delayed credit buffer aggregate drop is weak: disable_delayed_credit_buffer = 0.006.
5. hidden indirect causal path discovery remains unproven; candidate search appears useful, but not equivalent to causal path discovery.

## Recommendation

This PR is acceptable as a diagnostic branch, not as a solved structural necessity claim. It should remain open for review or be merged only with caveats preserved.

The next work item should be B6.3.1 Wrong-Trace and Mechanism Necessity Refinement, not B7.

