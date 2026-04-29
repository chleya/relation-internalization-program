from __future__ import annotations

import numpy as np

from .agents import BaseAgent
from .env import SlopeToyWorld, evaluate_chain
from .review import score_review_plan


def run_phase(agent: BaseAgent, env: SlopeToyWorld, n_steps: int, learn: bool = True) -> list[dict[str, object]]:
    records = []
    for _ in range(n_steps):
        context = env.sample_context()
        action = agent.act(context)
        result = env.step(action, context)
        state = result["state"]
        if learn:
            agent.observe(context, action, state.optimal_action, float(result["reward"]))
        records.append(
            {
                "context": context,
                "risk": state.risk,
                "optimal_action": state.optimal_action,
                "action": action,
                "reward": float(result["reward"]),
                "success": action == state.optimal_action,
            }
        )
    return records


def mean(records: list[dict[str, object]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def corrupt_monitoring_observation(
    context: dict[str, str],
    rng: np.random.Generator,
    noise_rate: float = 0.1,
) -> dict[str, str]:
    observed = dict(context)
    if rng.random() < noise_rate:
        observed["monitoring"] = "dense" if context["monitoring"] == "sparse" else "sparse"
    return observed


def noisy_observation_success(
    agent: BaseAgent,
    seed: int,
    n_steps: int = 200,
    noise_rate: float = 0.1,
) -> float:
    env = SlopeToyWorld(seed=seed, mode="ood")
    rng = np.random.default_rng(seed + 1000)
    successes = []
    for _ in range(n_steps):
        true_context = env.sample_context()
        observed_context = corrupt_monitoring_observation(true_context, rng, noise_rate=noise_rate)
        true_state = evaluate_chain(true_context)
        successes.append(agent.act(observed_context) == true_state.optimal_action)
    return sum(int(success) for success in successes) / len(successes)


def counterfactual_accuracy(agent: BaseAgent) -> float:
    cases = [
        (
            {"rainfall": "high", "drainage": "poor", "anchoring": "none", "toe_excavation": "no", "monitoring": "dense", "weather_label": "calm", "contractor_report": "normal"},
            "drain",
        ),
        (
            {"rainfall": "high", "drainage": "good", "anchoring": "none", "toe_excavation": "no", "monitoring": "dense", "weather_label": "alarm", "contractor_report": "warning"},
            "monitor",
        ),
        (
            {"rainfall": "low", "drainage": "good", "anchoring": "present", "toe_excavation": "yes", "monitoring": "sparse", "weather_label": "calm", "contractor_report": "normal"},
            "monitor",
        ),
    ]
    return sum(int(agent.act(context) == expected) for context, expected in cases) / len(cases)


def edit_success(agent: BaseAgent) -> float:
    context = {
        "rainfall": "high",
        "drainage": "good",
        "anchoring": "none",
        "toe_excavation": "no",
        "monitoring": "dense",
        "weather_label": "calm",
        "contractor_report": "normal",
    }
    before = agent.act(context)
    supported = agent.edit_link("Drainage -> PorePressureDown", False)
    after = agent.act(context)
    return float(supported and before == "monitor" and after == "drain")


def relation_audit(agent: BaseAgent) -> float:
    links = {item["link"]: item["enabled"] for item in agent.describe_relations()}
    required = [
        "Rainfall -> Infiltration",
        "Infiltration -> PorePressure",
        "Drainage -> PorePressureDown",
        "PorePressure -> Displacement",
        "Anchoring -> DisplacementDown",
        "ToeExcavation -> StabilityDown",
        "Displacement -> CrackExpansion",
        "CrackExpansion -> RiskUp",
    ]
    return sum(int(link in links) for link in required) / len(required)


def irrelevant_link_rejection(agent: BaseAgent) -> float:
    links = {item["link"]: item["enabled"] for item in agent.describe_relations()}
    irrelevant = [
        "WeatherLabel -> RiskUp",
        "ContractorReport -> RiskUp",
        "Drainage -> CrackDown",
    ]
    present = [link for link in irrelevant if link in links]
    if not present:
        return 1.0
    return sum(int(not links[link]) for link in present) / len(present)


def review_score(agent: BaseAgent) -> float:
    context = {
        "rainfall": "high",
        "drainage": "poor",
        "anchoring": "none",
        "toe_excavation": "no",
        "monitoring": "sparse",
        "weather_label": "calm",
        "contractor_report": "normal",
    }
    scored = score_review_plan(agent.review_plan(context))
    return float(scored["total"]) / float(scored["max_total"])


def review_consistency(agent: BaseAgent) -> float:
    cases = [
        (
            {
                "rainfall": "high",
                "drainage": "poor",
                "anchoring": "none",
                "toe_excavation": "no",
                "monitoring": "dense",
                "weather_label": "calm",
                "contractor_report": "normal",
            },
            "drain",
            "Drainage -> PorePressureDown",
        ),
        (
            {
                "rainfall": "low",
                "drainage": "good",
                "anchoring": "none",
                "toe_excavation": "yes",
                "monitoring": "sparse",
                "weather_label": "calm",
                "contractor_report": "normal",
            },
            "stop_work",
            "StopWork -> ExposureRiskDown",
        ),
        (
            {
                "rainfall": "low",
                "drainage": "good",
                "anchoring": "none",
                "toe_excavation": "yes",
                "monitoring": "dense",
                "weather_label": "calm",
                "contractor_report": "normal",
            },
            "anchor",
            "Anchoring -> DisplacementDown",
        ),
        (
            {
                "rainfall": "low",
                "drainage": "good",
                "anchoring": "present",
                "toe_excavation": "no",
                "monitoring": "dense",
                "weather_label": "alarm",
                "contractor_report": "warning",
            },
            "monitor",
            "Monitoring -> UncertaintyDown",
        ),
    ]
    passed = 0
    for context, expected_action, expected_point in cases:
        plan = agent.review_plan(context)
        passed += int(plan.get("action") == expected_action and plan.get("action_point") == expected_point)
    return passed / len(cases)


def gated_score(metrics: dict[str, float]) -> float:
    gates = [
        metrics["ood_success"] >= 0.8,
        metrics["spurious_attack_success"] >= 0.8,
        metrics["counterfactual_accuracy"] >= 0.8,
        metrics["edit_success"] >= 1.0,
        metrics["relation_audit"] >= 0.9,
        metrics["irrelevant_link_rejection"] >= 0.9,
        metrics["noisy_observation_success"] >= 0.7,
        metrics["review_score"] >= 0.9,
        metrics["review_consistency"] >= 0.9,
    ]
    if not all(gates):
        return 0.0
    return (
        0.2 * metrics["ood_success"]
        + 0.2 * metrics["spurious_attack_success"]
        + 0.16 * metrics["counterfactual_accuracy"]
        + 0.14 * metrics["edit_success"]
        + 0.1 * metrics["relation_audit"]
        + 0.05 * metrics["irrelevant_link_rejection"]
        + 0.08 * metrics["noisy_observation_success"]
        + 0.04 * metrics["review_score"]
        + 0.03 * metrics["review_consistency"]
    )


def evaluate_agent(agent: BaseAgent, seed: int, train_steps: int = 300, test_steps: int = 200) -> dict[str, float]:
    base_env = SlopeToyWorld(seed=seed, mode="base")
    base_records = run_phase(agent, base_env, train_steps, learn=True)
    ood_records = run_phase(agent, SlopeToyWorld(seed=seed + 1, mode="ood"), test_steps, learn=False)
    attack_records = run_phase(agent, SlopeToyWorld(seed=seed + 2, mode="spurious_attack"), test_steps, learn=False)
    metrics = {
        "base_reward": mean(base_records, "reward"),
        "base_success": mean(base_records, "success"),
        "ood_success": mean(ood_records, "success"),
        "spurious_attack_success": mean(attack_records, "success"),
        "counterfactual_accuracy": counterfactual_accuracy(agent),
        "edit_success": edit_success(agent),
        "relation_audit": relation_audit(agent),
        "irrelevant_link_rejection": irrelevant_link_rejection(agent),
        "noisy_observation_success": noisy_observation_success(agent, seed=seed + 3, n_steps=test_steps),
        "review_score": review_score(agent),
        "review_consistency": review_consistency(agent),
    }
    metrics["gated_slope_score"] = gated_score(metrics)
    return metrics
