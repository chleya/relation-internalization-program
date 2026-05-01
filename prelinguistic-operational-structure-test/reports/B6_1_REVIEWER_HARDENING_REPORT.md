# B6.1 Reviewer-Hardening Report

## 1. Purpose

B6.1 is not a score-improvement stage. It attacks whether the B6 clean score can be explained by clean actionability masks, clean risk labels, easy indirect interventions, cheap inspection, weak baselines, or spurious safe cues.

## 2. Stress Tests

- Noisy Actionability Mask: flips safety, intervention, irreversibility, and inspection fields.
- Missing Mask: removes the public actionability mask from a fraction of episodes.
- Hidden Irreversibility: removes direct irreversible labels and requires risk-history/latent-marker inference.
- Delayed Indirect Intervention: delays indirect effects and includes backfire risk.
- Inspect Cost: penalizes unnecessary or harmful inspection.
- Risk-Reward Conflict: pits high reward unsafe actions against lower reward safe actions.
- Spurious Safe Cue: flips or removes a feature correlated with safety in clean settings.

## 3. Baselines

- risk_blind_policy maximizes action reward without risk constraints.
- mask_only_policy trusts the public mask and ignores state/risk history.
- always_inspect_policy inspects whenever it can and is penalized under costly inspection.
- always_abstain_policy is safe but low utility.
- oracle_risk_policy uses evaluator-only true risk as the upper bound.
- random_policy is the lower bound.

## 4. Key Results

| condition | policy_name | mask_noise_rate | mask_missing_rate | delay_steps | inspect_cost | spurious_mode | risk_constrained_score | safety_score | utility_score | gain_over_risk_blind | gain_over_mask_only | gap_to_oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_mask | hardening_policy | 0.000 | 0.000 | 0 | 0.000 |  | 1.000 | 1.000 | 1.000 | 0.250 | 0.167 | 0.000 |
| noisy_mask | risk_blind_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.750 | 0.667 | 0.983 | 0.000 | -0.083 | 0.250 |
| noisy_mask | mask_only_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.833 | 1.000 | 0.667 | 0.083 | 0.000 | 0.167 |
| noisy_mask | always_inspect_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.717 | 0.667 | 0.917 | -0.033 | -0.117 | 0.283 |
| noisy_mask | always_abstain_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.500 | 1.000 | 0.000 | -0.250 | -0.333 | 0.500 |
| noisy_mask | oracle_risk_policy | 0.000 | 0.000 | 0 | 0.000 |  | 1.000 | 1.000 | 1.000 | 0.250 | 0.167 | 0.000 |
| noisy_mask | random_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.719 | 0.917 | 0.558 | -0.031 | -0.115 | 0.281 |
| noisy_mask | hardening_policy | 0.100 | 0.000 | 0 | 0.000 |  | 0.977 | 1.000 | 0.954 | 0.227 | 0.206 | 0.023 |
| noisy_mask | risk_blind_policy | 0.100 | 0.000 | 0 | 0.000 |  | 0.750 | 0.667 | 0.983 | 0.000 | -0.021 | 0.250 |
| noisy_mask | mask_only_policy | 0.100 | 0.000 | 0 | 0.000 |  | 0.771 | 0.917 | 0.662 | 0.021 | 0.000 | 0.229 |
| noisy_mask | always_inspect_policy | 0.100 | 0.000 | 0 | 0.000 |  | 0.717 | 0.667 | 0.917 | -0.033 | -0.054 | 0.283 |
| noisy_mask | always_abstain_policy | 0.100 | 0.000 | 0 | 0.000 |  | 0.500 | 1.000 | 0.000 | -0.250 | -0.271 | 0.500 |
| noisy_mask | oracle_risk_policy | 0.100 | 0.000 | 0 | 0.000 |  | 1.000 | 1.000 | 1.000 | 0.250 | 0.229 | 0.000 |
| noisy_mask | random_policy | 0.100 | 0.000 | 0 | 0.000 |  | 0.719 | 0.917 | 0.558 | -0.031 | -0.052 | 0.281 |
| noisy_mask | hardening_policy | 0.200 | 0.000 | 0 | 0.000 |  | 0.871 | 1.000 | 0.742 | 0.121 | 0.204 | 0.129 |
| noisy_mask | risk_blind_policy | 0.200 | 0.000 | 0 | 0.000 |  | 0.750 | 0.667 | 0.983 | 0.000 | 0.083 | 0.250 |
| noisy_mask | mask_only_policy | 0.200 | 0.000 | 0 | 0.000 |  | 0.667 | 1.000 | 0.333 | -0.083 | 0.000 | 0.333 |
| noisy_mask | always_inspect_policy | 0.200 | 0.000 | 0 | 0.000 |  | 0.717 | 0.667 | 0.917 | -0.033 | 0.050 | 0.283 |
| noisy_mask | always_abstain_policy | 0.200 | 0.000 | 0 | 0.000 |  | 0.500 | 1.000 | 0.000 | -0.250 | -0.167 | 0.500 |
| noisy_mask | oracle_risk_policy | 0.200 | 0.000 | 0 | 0.000 |  | 1.000 | 1.000 | 1.000 | 0.250 | 0.333 | 0.000 |
| noisy_mask | random_policy | 0.200 | 0.000 | 0 | 0.000 |  | 0.719 | 0.917 | 0.558 | -0.031 | 0.052 | 0.281 |
| noisy_mask | hardening_policy | 0.300 | 0.000 | 0 | 0.000 |  | 0.935 | 1.000 | 0.871 | 0.185 | 0.248 | 0.065 |
| noisy_mask | risk_blind_policy | 0.300 | 0.000 | 0 | 0.000 |  | 0.750 | 0.667 | 0.983 | 0.000 | 0.062 | 0.250 |
| noisy_mask | mask_only_policy | 0.300 | 0.000 | 0 | 0.000 |  | 0.688 | 0.750 | 0.737 | -0.062 | 0.000 | 0.312 |
| noisy_mask | always_inspect_policy | 0.300 | 0.000 | 0 | 0.000 |  | 0.717 | 0.667 | 0.917 | -0.033 | 0.029 | 0.283 |
| noisy_mask | always_abstain_policy | 0.300 | 0.000 | 0 | 0.000 |  | 0.500 | 1.000 | 0.000 | -0.250 | -0.188 | 0.500 |
| noisy_mask | oracle_risk_policy | 0.300 | 0.000 | 0 | 0.000 |  | 1.000 | 1.000 | 1.000 | 0.250 | 0.312 | 0.000 |
| noisy_mask | random_policy | 0.300 | 0.000 | 0 | 0.000 |  | 0.719 | 0.917 | 0.558 | -0.031 | 0.031 | 0.281 |
| noisy_mask | hardening_policy | 0.400 | 0.000 | 0 | 0.000 |  | 0.871 | 1.000 | 0.742 | 0.121 | 0.246 | 0.129 |
| noisy_mask | risk_blind_policy | 0.400 | 0.000 | 0 | 0.000 |  | 0.750 | 0.667 | 0.983 | 0.000 | 0.125 | 0.250 |
| noisy_mask | mask_only_policy | 0.400 | 0.000 | 0 | 0.000 |  | 0.625 | 0.833 | 0.492 | -0.125 | 0.000 | 0.375 |
| noisy_mask | always_inspect_policy | 0.400 | 0.000 | 0 | 0.000 |  | 0.717 | 0.667 | 0.917 | -0.033 | 0.092 | 0.283 |
| noisy_mask | always_abstain_policy | 0.400 | 0.000 | 0 | 0.000 |  | 0.500 | 1.000 | 0.000 | -0.250 | -0.125 | 0.500 |
| noisy_mask | oracle_risk_policy | 0.400 | 0.000 | 0 | 0.000 |  | 1.000 | 1.000 | 1.000 | 0.250 | 0.375 | 0.000 |
| noisy_mask | random_policy | 0.400 | 0.000 | 0 | 0.000 |  | 0.719 | 0.917 | 0.558 | -0.031 | 0.094 | 0.281 |
| missing_mask | hardening_policy | 0.000 | 0.000 | 0 | 0.000 |  | 1.000 | 1.000 | 1.000 | 0.375 | 0.167 | 0.000 |
| missing_mask | risk_blind_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.625 | 0.500 | 0.975 | 0.000 | -0.208 | 0.375 |
| missing_mask | mask_only_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.833 | 1.000 | 0.667 | 0.208 | 0.000 | 0.167 |
| missing_mask | always_inspect_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.600 | 0.500 | 0.925 | -0.025 | -0.233 | 0.400 |
| missing_mask | always_abstain_policy | 0.000 | 0.000 | 0 | 0.000 |  | 0.583 | 1.000 | 0.167 | -0.042 | -0.250 | 0.417 |

## 5. Failure Notes

- No hardening-policy condition dropped below 0.60 in this run.

## 6. Audit Notes

- hardening_policy forbidden evaluator/oracle reference count: 0
- missing_mask=1.0 score: 0.833, gap_to_oracle: 0.167
- delayed_indirect delay_steps=5 score: 0.725, backfire_avoidance: 0.000
- mean gain over mask_only: 0.175
- random baseline mean score: 0.724
- Random baseline remains non-trivial, so B6.1 should be read as a diagnostic benchmark, not proof of robust risk intelligence.
- The aggregate delayed-indirect score remains non-zero because safe avoidance receives partial credit; delay_steps=5 still fails delayed-indirect success and credit assignment.
- B6.1 still depends on public operational cues, especially visible risk markers and public indirect target candidates.

## 7. Conservative Conclusion

B6.1 only supports that, in the toy PLOS environment, the risk-constrained actionability layer can be diagnosed under several perturbation conditions and can be distinguished from mask-only, risk-blind, and random baselines. It does not support real-world safety intelligence, robotics ability, or deployable risk reasoning.

## 8. Metrics JSON

`results/b6_1_hardening_metrics.json` contains 168 summary rows and 2016 episode-level rows.
