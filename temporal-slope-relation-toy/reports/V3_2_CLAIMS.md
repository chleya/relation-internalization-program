# V3.2 Claims

Date: 2026-04-28

## 1. Stage Claim

V3.2 supports this narrow claim:

```text
In the toy temporal relation-chain setting, relation-specific takeover can be stress-tested under longer missing spans, correlated sensor failures, drift-like conflicts, multi-conflict audit, and delayed response cases.
```

## 2. Supported Claims

### 2.1 Long Missing Spans Are Covered

The stress test checks repeated missing `pore_pressure` observations across multiple time steps.

Current relation-chain agents score:

```text
long_missing_takeover_recall = 1.0
```

### 2.2 Correlated Sensor Failure Is Covered

The stress test checks simultaneous missing:

```text
displacement
crack
monitoring
```

Current relation-chain agents score:

```text
correlated_failure_recall = 1.0
```

### 2.3 Drift-Like Conflicts Are Covered

The stress test checks repeated conflicting `pore_pressure` observations.

Current relation-chain agents score:

```text
drift_conflict_takeover = 1.0
```

### 2.4 Multi-Conflict Audit Is Covered

The stress test requires audit to name multiple fields and multiple relation links.

Current relation-chain agents score:

```text
multi_conflict_audit_score = 1.0
```

### 2.5 Overcautious Takeover Still Fails

The `overcautious_takeover` negative control still fails:

```text
stress_v32_gated_score = 0.0
```

It catches hard takeover cases but fails audit specificity.

## 3. Not Supported

V3.2 does not support:

```text
real sensor reliability modeling
probabilistic calibration
human response timing
real engineering monitoring safety
deployment-ready takeover policy
```

## 4. Best One-Sentence Result

```text
V3.2 shows that the toy relation-chain takeover diagnostic can survive longer and more correlated partial-observability stress cases while still rejecting generic overcautious takeover.
```

