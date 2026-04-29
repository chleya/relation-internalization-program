# V3.2 Limitations

Date: 2026-04-28

## 1. Stress Cases Are Hand-Built

V3.2 stress cases are intentionally constructed to probe specific weaknesses.

They are not sampled from real monitoring distributions.

## 2. No Probabilistic Uncertainty Calibration

V3.2 uses discrete missing/noisy/conflict markers.

It does not estimate calibrated probabilities for:

```text
sensor failure
false positive
false negative
delayed observation reliability
```

## 3. No Human Response Model

Takeover is treated as an action label.

The project does not model:

```text
human response delay
human review quality
operator overload
responsibility transfer
```

## 4. Audit Still Uses Known Toy Links

Audit specificity is evaluated against predefined relation links.

This does not establish unrestricted relation discovery.

## 5. Perfect Scores Should Not Be Overread

The perfect stress scores mean the agents pass the current toy probes.

They do not imply real-world reliability.

## 6. Best Limitation Statement

```text
V3.2 strengthens the toy takeover diagnostic against longer and more correlated uncertainty cases, but it remains a synthetic stress test with predefined links and rule-defined takeover labels.
```

