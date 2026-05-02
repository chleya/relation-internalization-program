# B6.4 Second Pass Codex Prompt

```text
Current task: implement B6.4 Second Pass - Transfer Hardening.

Do not enter B7.
Do not add real robots.
Do not add real construction-site deployment.
Do not claim safety certification.
Do not claim engineering deployment.
Do not add 3D.
Do not add LLMs.
Do not add language tasks.
Do not expand the claim.

Stage:
B6.4 Second Pass - Transfer Hardening

Context:
B6.4 first pass created a transfer diagnostic harness, but several remaps remain shortcut-explainable.

Current first-pass result:
- b64_transfer_policy scores 1.000 on all remaps
- mean_baseline_transfer_gap = 0.169
- shortcut_equivalent_remap_count = 4
- visual_remap is matched by trace_only
- risk_cue_remap is matched by state_only
- dynamics_remap is matched by trace_only
- mask_visibility_remap is matched by state_only

Core goal:
Harden remaps so transfer cannot be explained by state_only, mask_only, trace_only, or trivial cue preservation.

Do not optimize for high mean score.
Optimize for transfer diagnostic validity and baseline separation.

Required hard remaps:

1. visual_remap_hard
   - remove direct public state target cue
   - alter spatial encoding
   - add distractor visual cues
   - test whether state_only and trace_only drop

2. risk_cue_remap_hard
   - invert, rotate, or re-encode risk cue
   - hide direct risk estimate
   - require feedback/history risk inference
   - test whether state_only drops

3. dynamics_remap_hard
   - alter velocity, friction, drift, and collision enough that shallow dynamics shortcuts fail
   - preserve latent operational structure
   - test trace/history/feedback transfer

4. mask_visibility_remap_hard
   - remove answer-like mask fields
   - hide unsafe/irreversible/cost/indirect target
   - test fallback inference
   - test mask_only drop

5. combined_remap_hard
   - combine visual + risk + dynamics + mask + delay remap
   - allow performance drop
   - measure whether policy still beats baselines

Required outputs:
- hard_transfer_score
- hard_transfer_drop
- hard_baseline_transfer_gap
- state_only_drop
- mask_only_drop
- trace_only_drop
- oracle_gap
- shortcut_equivalent_hard_remap_count
- transfer_evidence_strength
- remap_failure_reason

Required comparisons:
- b64_transfer_policy
- state_only
- mask_only
- trace_only
- random
- always_abstain
- conservative_uncertainty
- oracle

Required integrity checks:
- no evaluator_ground_truth access by policy
- no oracle_baseline_view access by policy
- no expected_decision access by policy
- no remap label used as answer key
- no hidden target exposed through metadata
- no empty split receives full score
- poisoned evaluator invariance passes

Claim boundary:
Even if second pass succeeds, it only supports toy-to-toy transfer evidence.
It does not support real-world generalization, robotics, safety certification, construction-site autonomy, or deployable engineering control.

Validation:
cd prelinguistic-operational-structure-test
python -m src.run_b6_4_transfer --config configs/b6_4_transfer_generalization.yaml --seed 0
python -m src.run_b6_4_result_review
python -m src.run_b6_4_adversarial_review
pytest -q
python -m src.visualize_b6_4 --summary results/b6_4_transfer_summary.csv
git diff --check

Final response must include:
1. Files changed.
2. Tests run.
3. Hard remaps implemented.
4. Hard transfer score summary.
5. Which hard remaps transfer.
6. Which hard remaps fail.
7. Baseline separation.
8. Any leakage or metric issue.
9. Whether B6.4 second pass is submit-ready as diagnostic branch.
10. Remaining blockers.
11. Confirm B7 was not implemented.
```
