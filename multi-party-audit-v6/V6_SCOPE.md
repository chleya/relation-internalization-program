# V6 Scope: Multi-Party Audit Resolution

Date: 2026-04-29

## 1. Why V6 Exists

V5 showed that a governance shell can log review outputs, block autonomous
approval, preserve replay, and keep responsibility traceable.

The next weakness is disagreement:

```text
What if two review outputs are both plausible but conflict?
```

V6 tests whether the system can preserve and audit disagreement instead of
collapsing it into automatic majority vote, confidence score, or compromise.

## 2. Core Question

```text
Can a resolver compare competing review outputs by relation-chain evidence,
uncertain links, takeover implications, and responsibility boundary, while
requiring human resolution for material disagreement?
```

## 3. What Counts As Success

A resolver must:

```text
detect disagreement
compare relation-chain evidence
preserve minority risk arguments
block automatic resolution
route material disagreement to human resolution
write replayable resolution logs
preserve responsibility boundary
```

## 4. What Fails

Resolvers fail if they:

```text
choose majority vote automatically
choose the highest confidence label only
average conflicting reviews into a compromise
drop minority risk evidence
hide disagreement from the log
claim approval authority
```

## 5. Boundary

V6 remains a toy diagnostic.

It does not support:

```text
real engineering arbitration
legal adjudication
expert replacement
deployment governance
```

