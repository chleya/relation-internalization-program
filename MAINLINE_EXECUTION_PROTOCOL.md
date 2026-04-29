# Mainline Execution Protocol

Date: 2026-04-28

## 1. Purpose

This protocol prevents the project from drifting into unrelated theory or product work.

The active mainline is:

```text
relation internalization diagnostics
```

The active stage is:

```text
V6 multi-party audit resolution
```

## 2. Work Cycle

Each cycle follows:

```text
1. state current claim
2. identify weakest false-positive explanation
3. implement one diagnostic or hardening test
4. run all old tests
5. run new sweep
6. write result and failure cases
7. update claim boundary
```

## 3. Definition Of Done

A stage is not complete until it has:

```text
working code
repeatable command
summary CSV
figures
tests
final report
self-audit
claim boundary
limitations
```

## 4. Adaptation Rules

If a result fails:

```text
record it
diagnose whether the model, metric, or environment failed
shrink the claim
decide whether to fix or freeze
```

Do not:

```text
hide failed metrics
weaken gates after seeing results without documenting why
rename failure as success
move to a new theory to avoid the failure
```

## 5. Mainline Filters

Before adding any task, ask:

```text
Does this distinguish relation internalization from prediction, memory, shortcut, probe readability, generic review, or unsafe automation?
```

If no, it is not mainline.

Before adding any engineering task, ask:

```text
Does this remain a toy diagnostic, or does it imply real safety deployment?
```

If it implies deployment, defer it.

## 6. Current Sprint Policy

Current sprint:

```text
Sprint 6: Freeze V6 multi-party audit resolution
```

Allowed work:

```text
V6 audit result
V6 claim boundary
V6 limitations
V6.1 hardening
fake evidence comparison rejection
hidden auto resolution rejection
minority risk preservation
tampered hash rejection
```

Disallowed work:

```text
real slope data
finite element modeling
NeuralSite integration
new symbol-emergence experiments
new communication-emergence experiments
general governance framework
real construction-plan approval
```
