# V5 Scope: Governance Shell

Date: 2026-04-29

## 1. Why V5 Exists

V1-V4 tested whether systems expose usable relations:

```text
static relations
neural probes and extraction
engineering-style relation chains
delayed temporal relations
uncertainty-aware takeover
bounded engineering review
```

The next weakness is workflow governance:

```text
Even if a review exposes relation chains and takeover conditions, can the system
record what happened, enforce approval gates, replay the decision, and preserve
human responsibility?
```

## 2. V5 Question

```text
Can a governance shell preserve relation-chain evidence, gate unsafe automation,
produce replayable logs, and maintain a responsibility chain?
```

## 3. What V5 Tests

V5 tests:

```text
audit log completeness
approval gate enforcement
takeover routing
replay consistency
responsibility traceability
non-deployment boundary preservation
```

## 4. What V5 Does Not Test

V5 does not test:

```text
real safety governance
legal responsibility
engineering standard compliance
production access control
real approval workflows
```

## 5. Minimal Governance Rule

The governance shell must block automatic approval if any of these are missing:

```text
relation_chain
uncertain_links
takeover_conditions for takeover_required cases
responsibility_boundary
claim_boundary
replay_hash
```

## 6. Negative Controls

V5 must reject:

```text
auto_approve_shell
no_log_shell
no_replay_shell
no_responsibility_shell
```

These may look operational but must fail the gated score.

