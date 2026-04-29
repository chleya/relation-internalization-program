from __future__ import annotations

from .agents_temporal import make_agent
from .agents_temporal import BaseTemporalAgent
from .env_temporal import TemporalStep, clone_with_rainfall, default_delay_map, evaluate_temporal_chain, generate_sequence, generate_temporal_sequence


def train_agent(agent: BaseTemporalAgent, seed: int, n_sequences: int = 200, mode: str = "base") -> None:
    for i in range(n_sequences):
        agent.observe_sequence(generate_sequence(seed + i, mode=mode))


def temporal_success(agent: BaseTemporalAgent, seed: int, n_sequences: int = 60, mode: str = "ood") -> float:
    total = 0
    correct = 0
    for i in range(n_sequences):
        seq = generate_sequence(seed + i, mode=mode)
        for t in range(len(seq)):
            correct += int(agent.act(seq, t) == seq[t].optimal_action)
            total += 1
    return correct / total


def delayed_counterfactual_accuracy(agent: BaseTemporalAgent) -> float:
    seq = generate_sequence(123, mode="ood")
    cases = []

    for t in range(0, len(seq) - 3):
        high_seq = clone_with_rainfall(seq, t, "high")
        low_seq = clone_with_rainfall(seq, t, "low")
        evaluate_temporal_chain(high_seq, rainfall_delay=1)
        evaluate_temporal_chain(low_seq, rainfall_delay=1)
        if high_seq[t + 1].pore_pressure != low_seq[t + 1].pore_pressure:
            cases.append((high_seq, low_seq, t + 1))

    if not cases:
        return 0.0
    correct = 0
    for high_seq, low_seq, target_t in cases:
        high_action = agent.act(high_seq, target_t)
        low_action = agent.act(low_seq, target_t)
        correct += int(high_action != low_action)
    return correct / len(cases)


def delay_edit_success(agent: BaseTemporalAgent) -> float:
    seq = [
        TemporalStep(0, "high", "poor", "none", "no", "dense", "normal", "calm", "normal"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "calm", "normal"),
        TemporalStep(2, "low", "poor", "none", "no", "dense", "normal", "calm", "normal"),
        TemporalStep(3, "low", "poor", "none", "no", "dense", "normal", "calm", "normal"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    before_t1 = agent.act(seq, 1)
    before_t2 = agent.act(seq, 2)
    if not agent.edit_delay("Rainfall -> PorePressure", 2):
        return 0.0
    after_t1 = agent.act(seq, 1)
    after_t2 = agent.act(seq, 2)
    return float(before_t1 == "drain" and before_t2 != "drain" and after_t1 != "drain" and after_t2 == "drain")


def temporal_audit_score(agent: BaseTemporalAgent) -> float:
    audit = agent.temporal_audit()
    required = [
        "Rainfall[t-1] -> PorePressure[t]",
        "PorePressure[t-1] -> Displacement[t]",
        "Displacement[t-1] + Monitoring[t-1] -> Crack[t]",
    ]
    return sum(int(item in audit) for item in required) / len(required)


def gated_temporal_score(metrics: dict[str, float]) -> float:
    gates = [
        metrics["temporal_ood_success"] >= 0.8,
        metrics["delayed_counterfactual_accuracy"] >= 0.8,
        metrics["delay_edit_success"] >= 0.9,
        metrics["surface_shortcut_rejection"] >= 0.8,
        metrics["temporal_audit_score"] >= 0.9,
    ]
    if not all(gates):
        return 0.0
    return (
        0.25 * metrics["temporal_ood_success"]
        + 0.25 * metrics["delayed_counterfactual_accuracy"]
        + 0.2 * metrics["delay_edit_success"]
        + 0.15 * metrics["surface_shortcut_rejection"]
        + 0.15 * metrics["temporal_audit_score"]
    )


def trained_clone(agent: BaseTemporalAgent, seed: int) -> BaseTemporalAgent:
    clone = make_agent(agent.name)
    train_agent(clone, seed=seed, mode="base")
    return clone


def evaluate_agent(agent: BaseTemporalAgent, seed: int) -> dict[str, float]:
    base_agent = trained_clone(agent, seed)
    metrics = {
        "temporal_ood_success": temporal_success(base_agent, seed + 100, mode="ood"),
        "surface_shortcut_rejection": temporal_success(trained_clone(agent, seed), seed + 200, mode="spurious_attack"),
        "delayed_counterfactual_accuracy": delayed_counterfactual_accuracy(trained_clone(agent, seed)),
        "delay_edit_success": delay_edit_success(trained_clone(agent, seed)),
        "temporal_audit_score": temporal_audit_score(trained_clone(agent, seed)),
    }
    metrics["gated_temporal_score"] = gated_temporal_score(metrics)
    return metrics


def _train_with_delay(agent: BaseTemporalAgent, seed: int, delay: int, n_sequences: int, length: int, correlation: float = 0.9) -> None:
    delay_map = default_delay_map(delay)
    for i in range(n_sequences):
        seq = generate_temporal_sequence(
            seed + i,
            length=length,
            delay_map=delay_map,
            shortcut_correlation=correlation,
            mode="hardening",
        )
        agent.observe_sequence(seq)


def _accuracy_on_delay(agent: BaseTemporalAgent, seed: int, delay: int, n_sequences: int, length: int, correlation: float = 0.1) -> float:
    total = 0
    correct = 0
    delay_map = default_delay_map(delay)
    for i in range(n_sequences):
        eval_agent = make_agent(agent.name)
        if hasattr(eval_agent, "delay_map"):
            eval_agent.delay_map = delay_map.copy()
        seq = generate_temporal_sequence(
            seed + i,
            length=length,
            delay_map=delay_map,
            shortcut_correlation=correlation,
            mode="hardening",
        )
        if agent.name == "learned_delayed_links":
            eval_agent.observe_sequence(seq)
        elif agent.name not in {"delayed_relation_chain"}:
            eval_agent = agent
        for t in range(len(seq)):
            correct += int(eval_agent.act(seq, t) == seq[t].optimal_action)
            total += 1
    return correct / total


def variable_delay_success(agent: BaseTemporalAgent, seed: int, config: dict) -> float:
    length = int(config["sequence"]["length"])
    n_train = int(config["sequence"]["n_train_sequences"])
    n_test = int(config["sequence"]["n_test_sequences"])
    for delay in [1, 2]:
        _train_with_delay(agent, seed + delay * 1000, delay, n_train // 3, length)
    heldout = int(config["variable_delay"]["heldout_delay"])
    return _accuracy_on_delay(agent, seed + 5000, heldout, n_test, length)


def false_delay_shortcut_rejection(agent: BaseTemporalAgent, seed: int, config: dict) -> float:
    length = int(config["sequence"]["length"])
    n_train = int(config["sequence"]["n_train_sequences"])
    n_test = int(config["sequence"]["n_test_sequences"])
    train_corr = float(config["false_shortcut"]["train_correlation"])
    test_corr = float(config["false_shortcut"]["test_correlation"])
    _train_with_delay(agent, seed + 7000, 1, n_train, length, correlation=train_corr)
    return _accuracy_on_delay(agent, seed + 8000, 1, n_test, length, correlation=test_corr)


def multi_link_delay_edit_success(agent: BaseTemporalAgent, config: dict) -> float:
    links = config["delay_edit"]["links"]
    targets = [(int(pair[0]), int(pair[1])) for pair in config["delay_edit"]["edit_targets"]]
    total = 0
    success = 0
    for link in links:
        for old_delay, new_delay in targets:
            delay_map = default_delay_map(1)
            if link not in delay_map:
                continue
            delay_map[link] = old_delay
            local_agent = make_agent(agent.name)
            if hasattr(local_agent, "delay_map"):
                local_agent.delay_map = delay_map.copy()
            edited = local_agent.edit_delay(link, new_delay)
            total += 1
            if edited and getattr(local_agent, "delay_map", {}).get(link) == new_delay:
                success += 1
    return success / total if total else 0.0


def temporal_audit_consistency(agent: BaseTemporalAgent, expected_delay_map: dict[str, int]) -> float:
    if hasattr(agent, "delay_map"):
        agent.delay_map = expected_delay_map.copy()
    audit = agent.audit_temporal_relations()
    expected = [
        f"Rainfall[t-{expected_delay_map['Rainfall -> PorePressure']}] -> PorePressure[t]",
        f"PorePressure[t-{expected_delay_map['PorePressure -> Displacement']}] -> Displacement[t]",
        f"Displacement[t-{expected_delay_map['Displacement -> Crack']}] + Monitoring[t-{expected_delay_map['Displacement -> Crack']}] -> Crack[t]",
        f"Drainage[t-{expected_delay_map['Drainage -> PorePressureDown']}] -> PorePressureDown[t]",
        f"Anchoring[t-{expected_delay_map['Anchoring -> DisplacementDown']}] -> DisplacementDown[t]",
    ]
    return sum(int(item in audit and "[t-" in item) for item in expected) / len(expected)


def anti_template_generalization(agent: BaseTemporalAgent, seed: int, config: dict) -> float:
    length = int(config["sequence"]["length"])
    n_train = int(config["sequence"]["n_train_sequences"])
    n_test = int(config["sequence"]["n_test_sequences"])
    for i, delay in enumerate([1, 2, 3]):
        _train_with_delay(agent, seed + 9000 + i * 1000, delay, n_train // 3, length)
    scores = [_accuracy_on_delay(agent, seed + 12000 + delay * 1000, delay, n_test // 3, length) for delay in [1, 2, 3]]
    return sum(scores) / len(scores)


def hardening_gated_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["variable_delay_success"] >= gates["variable_delay_success"],
        metrics["false_delay_shortcut_rejection"] >= gates["false_delay_shortcut_rejection"],
        metrics["multi_link_delay_edit_success"] >= gates["multi_link_delay_edit_success"],
        metrics["temporal_audit_consistency"] >= gates["temporal_audit_consistency"],
        metrics["anti_template_generalization"] >= gates["anti_template_generalization"],
    ]
    if not all(required):
        return 0.0
    return (
        0.25 * metrics["variable_delay_success"]
        + 0.20 * metrics["false_delay_shortcut_rejection"]
        + 0.25 * metrics["multi_link_delay_edit_success"]
        + 0.15 * metrics["temporal_audit_consistency"]
        + 0.15 * metrics["anti_template_generalization"]
    )


def evaluate_hardening_agent(agent: BaseTemporalAgent, seed: int, config: dict) -> dict[str, float]:
    metrics = {
        "variable_delay_success": variable_delay_success(make_agent(agent.name), seed, config),
        "false_delay_shortcut_rejection": false_delay_shortcut_rejection(make_agent(agent.name), seed, config),
        "multi_link_delay_edit_success": multi_link_delay_edit_success(make_agent(agent.name), config),
        "temporal_audit_consistency": temporal_audit_consistency(make_agent(agent.name), default_delay_map(1)),
        "anti_template_generalization": anti_template_generalization(make_agent(agent.name), seed, config),
    }
    metrics["hardening_gated_score"] = hardening_gated_score(metrics, config["gates"])
    return metrics
