# G1.2 Feature Induction Adversarial Review

## Decision
- decision: submit_ready_as_toy_feature_induction_diagnostic
- leakage_issue_found: true
- metric_issue_found: false

## Score Audit
- g1_2_mean_score = 0.813
- g1_2_ood_score = 0.887
- feedback_feature_drop = 0.198
- compression_feature_drop = 0.308
- gain_over_random = 0.611
- oracle_gap = 0.085
- mask_f1 = 0.750

## Feature Usage Audit
- compression_surprise: used
- delay_signal: UNUSED
- feedback_success: used
- indirect_evidence: UNUSED
- intervention_gain: UNUSED
- prediction_error: UNUSED
- risk_inverse: UNUSED

## Selected Program
- direct_features: ['feedback_success']
- indirect_features: ['compression_surprise']
- inspect_features: ['compression_surprise']
- direct_threshold: 0.45
- indirect_threshold: 0.65
- inspect_threshold: 0.45
- risk_threshold: 0.62

## Remaining Blockers
- generator source contains forbidden evaluator/oracle references
- feature vocabulary is still hand-defined (7 names)
- OOD remap is synthetic noise shift, not truly new world dynamics

## Recommended Next Step
G1.3 expand feature vocabulary (unsupervised feature discovery vs hand-named features) before G2 compositional scaling
