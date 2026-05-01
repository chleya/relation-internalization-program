# B4.1 Self-Audit

## What This Improves

- Investigates B4 high-score risk.
- Checks per-episode action overlap.
- Checks fixed action-type shortcut.
- Checks action policy provenance.
- Adds family-specific action scorers.
- Adds shared action policy ablation.
- Adds wrong-action / wrong-region stress.
- Re-checks baseline sanity.
- Adds action-after-trace-ablation specificity.
- Adds intervention-value leakage check.

## Remaining Weaknesses

- Still a 64x64 toy world.
- Action space remains hand-designed.
- Intervention values are simulator-defined.
- Wrong-action stress is synthetic.
- Provenance fields may miss hidden shared logic.
- Ablation may create OOD internal states.
- Passing B4.1 does not imply real control.

## False Positive Risks

- Different action scorers may share low-level features.
- Action overlap may be low due to noise, not real mechanism separation.
- Wrong-action penalty may be evaluator-biased.
- Oracle intervention values may encode generator assumptions.
- Trace ablation may damage general action capacity.
- Fixed action-type shortcut may persist in subtle form.

## Required Failure Checks

1. three models choose same action per episode
2. action_type is almost always fixed
3. shared action policy is used
4. private trace action scorer is not used
5. action scorer correlations are too high
6. shared action policy ablation causes large drop
7. wrong action / wrong region is not penalized
8. random / saliency / short-horizon / inspect-only baselines match trace model
9. private trace ablation is not stronger than non-trace ablation
10. intervention value leakage is detected
