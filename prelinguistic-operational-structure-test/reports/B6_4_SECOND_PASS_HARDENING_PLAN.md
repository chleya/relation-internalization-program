# B6.4 Second-Pass Hardening Plan

## Purpose

B6.4 first pass is a transfer harness pass, not strong transfer evidence. The second pass should harden shortcut-equivalent remaps while keeping the task toy-to-toy.

## 1. visual_remap_hard

- remove direct public state target cue
- alter spatial encoding
- introduce appearance distractors
- test state_only drop

## 2. risk_cue_remap_hard

- invert or rotate risk cue encoding
- hide direct risk estimate
- require feedback/history risk inference
- test risk cue transfer

## 3. dynamics_remap_hard

- alter dynamics enough that short-horizon, trace-only, and conservative rules fail
- preserve latent operational structure
- test trace/history contribution

## 4. mask_visibility_remap_hard

- remove answer-like mask fields
- hide unsafe/irreversible/cost/indirect target
- test fallback inference and history

## 5. combined_remap_hard

- combine visual, risk, dynamics, mask, and delay remaps
- allow expected score drop
- measure relative baseline gap

## 6. Anti-Overfit Criteria

- transfer evidence requires b64_transfer_policy to beat state_only, mask_only, and trace_only by a meaningful margin
- shortcut baselines must degrade under the remap they are supposed to fail
- oracle_gap must remain interpretable
- high b64 score alone is insufficient
- no real-world risk intelligence, robotics, safety certification, construction-site autonomy, or deployable control claim is allowed
