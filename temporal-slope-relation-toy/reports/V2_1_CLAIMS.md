# V2.1 Claims

Date: 2026-04-28

## Supported

1. The temporal diagnostic distinguishes delayed relation-chain use from temporal prediction.
2. Structural memory can score high on temporal prediction but fails editable/auditable relation-chain gates.
3. Learned delayed links can pass variable delay, false shortcut rejection, multi-link edit, and time-indexed audit in the toy setting.
4. V2.1 reduces fixed-template and single-link false positives compared with V2.

## Not Supported

1. Real geotechnical time-series modeling.
2. Unrestricted temporal relation discovery.
3. Safety-calibrated slope monitoring.
4. Deployment-ready engineering review AI.
5. General proof that large AI systems internalize real-world engineering relations.

## Current Core Claim

```text
Temporal prediction is not temporal relation internalization.
```

## Evidence

Mean over seeds 0-4:

```text
structural_memory_temporal:
variable_delay_success = 0.985
false_delay_shortcut_rejection = 0.980
anti_template_generalization = 0.975
multi_link_delay_edit_success = 0.000
temporal_audit_consistency = 0.000
hardening_gated_score = 0.000
```

This is the central negative control.
