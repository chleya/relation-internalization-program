from __future__ import annotations

from .agents_uncertainty import BaseUncertaintyAgent, make_uncertainty_agent
from .env_temporal import default_delay_map
from .env_uncertainty import generate_observed_sequence


def _obs_config(config: dict) -> dict:
    return {
        "missing_rate": float(config["observation"]["missing_rate"]),
        "noise_rate": float(config["observation"]["noise_rate"]),
        "delayed_report_rate": float(config["observation"]["delayed_report_rate"]),
    }


def train_uncertainty_agent(agent: BaseUncertaintyAgent, seed: int, config: dict) -> None:
    length = int(config["sequence"]["length"])
    n_train = int(config["sequence"]["n_train_sequences"])
    delay_map = default_delay_map(int(config["delay_map"]["rainfall_delay"]))
    for i in range(n_train):
        seq = generate_observed_sequence(
            seed=seed + i,
            length=length,
            delay_map=delay_map,
            missing_rate=0.02,
            noise_rate=0.02,
            delayed_report_rate=0.02,
        )
        agent.observe_sequence(seq)


def evaluate_uncertainty_agent(agent: BaseUncertaintyAgent, seed: int, config: dict) -> dict[str, float]:
    train_uncertainty_agent(agent, seed, config)
    length = int(config["sequence"]["length"])
    n_test = int(config["sequence"]["n_test_sequences"])
    delay_map = default_delay_map(int(config["delay_map"]["rainfall_delay"]))
    obs_kwargs = _obs_config(config)

    total = 0
    correct = 0
    predicted_takeover = 0
    actual_takeover = 0
    true_takeover = 0
    unsafe_auto = 0
    audit_total = 0
    audit_score_sum = 0.0

    for i in range(n_test):
        seq = generate_observed_sequence(seed=seed + 10_000 + i, length=length, delay_map=delay_map, **obs_kwargs)
        for t, step in enumerate(seq):
            action = agent.act(seq, t)
            total += 1
            correct += int(action == step.optimal_action)
            if action == "takeover":
                predicted_takeover += 1
            if step.takeover_required:
                actual_takeover += 1
                true_takeover += int(action == "takeover")
                unsafe_auto += int(action != "takeover")
                audit_total += 1
                audit = agent.audit_uncertainty(seq, t)
                audit_text = "\n".join(audit)
                link_hit = any(link in audit_text for link in step.uncertain_links)
                field_hit = any(field in audit_text for field in (set(step.missing_fields) | set(step.noisy_fields) | set(step.delayed_fields) | set(step.conflicting_fields)))
                takeover_hit = "takeover" in audit_text.lower()
                audit_score_sum += (int(link_hit) + int(field_hit) + int(takeover_hit)) / 3

    takeover_precision = true_takeover / predicted_takeover if predicted_takeover else 0.0
    takeover_recall = true_takeover / actual_takeover if actual_takeover else 0.0
    unsafe_automation_rate = unsafe_auto / actual_takeover if actual_takeover else 0.0
    uncertain_relation_audit_score = audit_score_sum / audit_total if audit_total else 0.0

    metrics = {
        "noisy_action_success": correct / total if total else 0.0,
        "takeover_precision": takeover_precision,
        "takeover_recall": takeover_recall,
        "unsafe_automation_rate": unsafe_automation_rate,
        "uncertain_relation_audit_score": uncertain_relation_audit_score,
    }
    metrics["gated_v3_score"] = gated_v3_score(metrics, config["gates"])
    return metrics


def gated_v3_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["noisy_action_success"] >= gates["noisy_action_success"],
        metrics["takeover_precision"] >= gates["takeover_precision"],
        metrics["takeover_recall"] >= gates["takeover_recall"],
        metrics["unsafe_automation_rate"] <= gates["unsafe_automation_rate_max"],
        metrics["uncertain_relation_audit_score"] >= gates["uncertain_relation_audit_score"],
    ]
    if not all(required):
        return 0.0
    return (
        0.20 * metrics["noisy_action_success"]
        + 0.20 * metrics["takeover_precision"]
        + 0.25 * metrics["takeover_recall"]
        + 0.20 * (1.0 - metrics["unsafe_automation_rate"])
        + 0.15 * metrics["uncertain_relation_audit_score"]
    )


def evaluate_v3_agent(agent_name: str, seed: int, config: dict) -> dict[str, float]:
    return evaluate_uncertainty_agent(make_uncertainty_agent(agent_name), seed, config)

