# V3.1 Limitations

Date: 2026-04-28

## 1. Hand-Constructed Hardening Cases

V3.1 hardening cases are deliberately small and hand-constructed.

This is useful for precise failure probes, but it does not replace larger stochastic stress testing.

## 2. String-Based Audit Scoring

Audit specificity is measured by checking whether audit text contains:

```text
takeover
field name
relation link name
```

This catches generic audit, but it is still a string-level proxy.

## 3. Predefined Relation Links

The links remain predefined:

```text
Rainfall -> PorePressure
PorePressure -> Displacement
Displacement -> Crack
Drainage -> PorePressureDown
Anchoring -> DisplacementDown
```

V3.1 does not test unrestricted discovery.

## 4. Toy Takeover Semantics

The distinction between benign uncertainty and takeover-required uncertainty is rule-defined.

It has not been validated by domain experts.

## 5. Limited Noise Types

V3.1 tests:

```text
irrelevant missing sensor
benign noise
overuse
conflicting evidence
generic audit
```

It does not yet test:

```text
correlated sensor failure
adversarial missingness
long missing spans
multiple simultaneous conflicts
calibration drift
human delayed response
```

## 6. Best Limitation Statement

```text
V3.1 hardens the toy takeover diagnostic against generic overcaution and template audit, but it remains a controlled toy test rather than a validated engineering takeover policy.
```

