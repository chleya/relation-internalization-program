# V3 Charter

Date: 2026-04-28

## Purpose

V3 tests uncertainty-aware relation use.

V3 asks:

```text
When observations are missing, noisy, or delayed, can the system maintain a relation chain and trigger human takeover when automatic action is unsafe?
```

## Core Additions

- Missing sensors.
- Delayed noisy observations.
- `takeover` action.
- Takeover precision and recall.
- Unsafe automation rate.
- Uncertain relation-chain audit.

## Non-Goals

- Do not use real slope data.
- Do not claim real engineering safety capability.
- Do not build a numerical slope model.
- Do not expand unrestricted relation discovery.

## Expected Core Metrics

```text
noisy_action_success
takeover_precision
takeover_recall
unsafe_automation_rate
uncertain_relation_audit
gated_v3_score
```

## Central V3 Claim To Test

```text
Temporal relation internalization is not enough; engineering-grade relation use requires uncertainty-aware takeover.
```
