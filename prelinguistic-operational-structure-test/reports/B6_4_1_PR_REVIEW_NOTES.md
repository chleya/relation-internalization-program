# B6.4.1 PR Review Notes

## PR Status

- PR: #5
- URL: https://github.com/chleya/relation-internalization-program/pull/5
- Branch: b6-4-1-transfer-hardening
- Commit: f893de7
- Status: OPEN

## What This PR Adds

- B6.4.1 Transfer Hardening diagnostics
- visual_remap_hard
- risk_cue_remap_hard
- dynamics_remap_hard
- mask_visibility_remap_hard
- combined_remap_hard
- hard-transfer metrics
- baseline drop checks
- adversarial result review

## Validation

- hard_transfer_score = 0.935
- hard_baseline_transfer_gap = 0.605
- shortcut_equivalent_hard_remap_count = 0
- b6_4_1_submit_ready_as_diagnostic = true
- b6_4_1_hidden_answer_cue_count = 0
- pytest -q: 377 passed
- git diff --check = clean

## Supported Evidence

- B6.4.1 is stronger than B6.4 first pass.
- Shortcut-equivalent hard remap count drops to 0.
- Hard remaps separate b64_1 policy from state_only, mask_only, and trace_only.
- No hidden answer cue or evaluator/oracle leakage was found.

## Remaining Limitation

combined_remap_hard remains limited:

- policy = 0.675
- oracle = 1.000
- oracle gap = 0.325

This is the main remaining hard-transfer blocker.

## Claim Boundary

B6.4.1 supports only toy-to-toy hard-remap diagnostic evidence.

It does not support:

- real-world transfer
- robotics capability
- safety certification
- construction-site autonomy
- deployable engineering control
- B7 readiness

## Recommendation

Keep PR #5 open for review or merge only with the combined_remap_hard caveat preserved. The next work item should be B6.4.2 Combined-Remap Hardening, not B7.
