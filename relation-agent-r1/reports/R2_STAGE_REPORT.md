# R2 Stage Report: Partial Observability Active Relation Agent

Date: 2026-04-29

## 1. Executive Claim

R2 shows that relation discovery alone is not sufficient under partial observability.

The R1.2 discovery agent can recover useful process links and achieve high action success when the state is observable. Under missing, noisy, or conflicting observations, however, it still acts as if the relation chain is verified. That behavior produces a zero gated score in R2 because the agent fails the safety-relevant requirements: inspection recall, unsafe automation control, and uncertainty audit.

The key result is therefore not only that the uncertainty-aware agent passes. The stronger result is that a relation-discovery agent without uncertainty handling fails even when its apparent task success remains high.

## 2. What R2 Adds

R2 extends the R1 line from relation use to relation use under epistemic uncertainty.

It introduces three conditions absent from earlier stages:

- Missing critical observations: `pore_pressure`, `displacement`, and `risk` may be `unknown`.
- Noisy or conflicting observations: observed fields can contradict learned process relations, such as `displacement=normal` with `risk=high`.
- Unsafe automation accounting: acting directly from an unverifiable relation chain is counted as unsafe automation, even if the action sometimes happens to match the true optimal action.

This changes the evaluation target. A passing agent must not merely select good actions. It must know when its action basis is not currently justified and inspect before automating.

## 3. Stage Results

| agent | partial_observation_success | inspection_recall | unsafe_action_rate | uncertainty_audit_score | noisy_observation_robustness | partial_r2_gated_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 0.433 | 0.160 | 0.840 | 0.000 | 0.325 | 0.000 |
| shortcut | 0.893 | 0.000 | 1.000 | 0.000 | 0.470 | 0.000 |
| passive_memory | 0.772 | 0.000 | 1.000 | 0.000 | 0.412 | 0.000 |
| discovery_relation_agent | 0.967 | 0.000 | 1.000 | 0.000 | 0.448 | 0.000 |
| uncertainty_discovery_agent | 1.000 | 1.000 | 0.000 | 1.000 | 0.912 | 0.982 |

The most important contrast is between `discovery_relation_agent` and `uncertainty_discovery_agent`.

`discovery_relation_agent` reaches `0.967` partial observation success, but still receives a gated score of `0.000`. It has no inspection recall, always automates under critical uncertainty, and provides no relation-specific uncertainty audit.

`uncertainty_discovery_agent` preserves action performance while adding inspection and audit behavior. Its gated score is `0.982`, with zero unsafe automation in the tested critical-missing cases.

## 4. Interpretation

R2 separates three abilities that can look similar in a simple action-success metric:

- Acting from discovered relations.
- Recognizing that a relation chain is not currently observable.
- Choosing an information-gathering action before committing to automation.

The R1.2 discovery agent has the first ability but not the second or third. This is why R2 is a meaningful stage rather than a minor robustness test. It exposes a failure mode that would be hidden if the benchmark only asked whether final actions are often correct.

In this toy world, an agent can be operationally effective and still epistemically unsafe. R2 makes that distinction explicit.

## 5. Why The Gated Score Matters

The gated score is intentionally harsh. It prevents a high average action score from masking failure on safety-relevant subtests.

This is visible in the R2 result:

- `shortcut` has high partial observation success but no inspection behavior.
- `passive_memory` can sometimes act correctly from memorized patterns but cannot audit uncertainty.
- `discovery_relation_agent` discovers useful relations but treats missing or contradictory observations as actionable.
- Only `uncertainty_discovery_agent` passes the combined requirements.

The gate therefore enforces the intended claim boundary: R2 is not testing whether an agent can guess well under incomplete input. It is testing whether the agent can stop, inspect, and name the uncertain relation basis before acting.

## 6. Current Boundary

R2 remains a toy diagnostic.

It does not show:

- Real slope safety capability.
- Deployment readiness.
- General causal discovery.
- Learned uncertainty calibration from field feedback.
- Robustness to adversarial missingness patterns.
- Cost-sensitive inspection policies.

The current uncertainty behavior is still partly rule-shaped. `inspect` also reveals the true state immediately, which makes the information action too clean compared with real monitoring.

## 7. R2.1 Direction

R2.1 should harden partial observability against a trivial template:

```text
if anything is missing: inspect
```

That template can pass some R2 cases without genuinely using relation structure. The next stage should therefore distinguish relation-specific uncertainty from blanket conservatism.

Recommended R2.1 tests:

- Non-critical missingness: irrelevant or action-independent unknown fields should not force inspection.
- Inspection budget pressure: the agent should reserve inspection for relation-critical uncertainty.
- Mixed observability cases: some missing fields are irrelevant because other relation paths already determine the safe action.
- Conflict localization: the audit should identify the specific inconsistent link, not merely report that the state is incomplete.
- Delayed or noisy inspection: inspection should reduce uncertainty but not act as an oracle.
- Comparative baseline: include a `missing_always_inspect` policy and require the uncertainty agent to outperform it under cost and precision metrics.

The central R2.1 question should be:

```text
Can the agent inspect because the relation chain is unverifiable, not merely because a field is unknown?
```

## 8. Provisional Conclusion

The R1 sequence now supports a narrower and stronger claim than the original R1 claim:

```text
A small non-LLM agent can learn usable process relations through interaction,
use them for action, discover relations from transition evidence, and improve
partial-observability behavior when uncertainty over relation links is made
explicit.
```

The claim still excludes real engineering competence. But R2 adds an important internal criterion: a relation agent must represent not only what relations it believes, but also when those relations cannot be verified from the current observation.
