# B6.2 Fallback Risk Inference / Delayed Credit Hardening

## Purpose

B6.2 addresses audit-derived structural issues before any higher-stage claim: missing masks, delayed indirect credit, wrong trace regions, reduced mask visibility, poisoned evaluator invariance, and stronger baselines.

## Validation

Local validation commands:

```bash
cd prelinguistic-operational-structure-test
pytest -q
python -m src.run_b6_2_hardening --config configs/b6_2_hardening.yaml --seed 0
python -m src.visualize_b6_2 --summary results/b6_2_hardening_summary.csv
git diff --check
```

## Key Results

- b62_policy_mean_score = 0.976
- oracle_mean_score = 1.000
- random_mean_score = 0.561

## Second-Pass Caveats

B6.2 is a diagnostic layer, not a completed robustness claim. The aggregate mean can hide split-level failures, so B6.2 must be interpreted together with `reports/B6_2_RESULT_REVIEW.md`.

- wrong_trace is the key unresolved weakness: if b62_policy does not outperform mask_only or state_only, B6.2 does not prove autonomous target correction or robust trace repair.
- hide_indirect_target tests candidate-search fallback. It does not prove hidden indirect causal path discovery unless the policy succeeds without public indirect cues and without oracle target leakage.
- missing_mask improvements are fallback diagnostics. If they depend mostly on public state estimates rather than private trace, the claim remains narrow.
- delayed_indirect improvements are toy outcome-history diagnostics. A high aggregate score can still coexist with weak delayed indirect success if safety or abstention receives partial credit.
- mask_only and state_only baselines must remain visible in the interpretation. If they explain most performance, B6.2 remains partly a mask/state diagnostic benchmark.

## Audit-Derived Limitations

1. B6/B6.1 clean mask exposed strong operational cues.
2. previous_trace_state["region"] remains a strong target prior, now explicitly stressed through wrong/missing/ambiguous trace conditions.
3. B6 clean config had n_ood but no active OOD split; B6.2 uses explicit condition-level stress splits instead.
4. Some trace-uncertainty modes may be represented inside a wrong_trace split rather than as standalone missing_trace / ambiguous_trace / low_confidence_trace splits; they must not be reported as independently passed unless generated separately.
5. Hidden indirect causal path discovery is not proven.
6. Delayed credit assignment improvements, if any, are only in toy delayed outcome history.
7. If mask_only remains strong, B6.2 remains partly a mask-diagnostic task.
8. If random/always_abstain remain non-trivial, conservative default behavior still earns partial score.
9. B6.2 does not support real-world risk intelligence, safety certification, robotics ability, or engineering deployment.
