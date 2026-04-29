from __future__ import annotations

from .agents_uncertainty import BaseUncertaintyAgent, make_uncertainty_agent
from .env_temporal import TemporalStep, evaluate_temporal_chain
from .env_uncertainty import ObservedTemporalStep, apply_observation_uncertainty, finalize_observed_sequence, finalize_observed_step


def _clean_observed(seq: list[TemporalStep]) -> list[ObservedTemporalStep]:
    import numpy as np

    return apply_observation_uncertainty(seq, np.random.default_rng(0), 0.0, 0.0, 0.0)


def _high_rain_sequence(length: int = 6) -> list[ObservedTemporalStep]:
    seq = [
        TemporalStep(t, "high" if t == 0 else "low", "poor", "none", "no", "dense", "normal", "clear", "safe")
        for t in range(length)
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    return _clean_observed(seq)


def _toe_excavation_sequence(length: int = 6) -> list[ObservedTemporalStep]:
    seq = [
        TemporalStep(t, "low", "poor", "none", "yes" if t == 0 else "no", "dense", "normal", "clear", "safe")
        for t in range(length)
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    return _clean_observed(seq)


def _finalize_many(seq: list[ObservedTemporalStep], indices: list[int]) -> list[ObservedTemporalStep]:
    for idx in indices:
        finalize_observed_step(seq[idx])
    finalize_observed_sequence(seq)
    return seq


def long_missing_cases() -> list[tuple[list[ObservedTemporalStep], int]]:
    seq = _high_rain_sequence()
    for t in [1, 2, 3]:
        seq[t].pore_pressure = "unknown"
        seq[t].missing_fields = tuple(sorted(set(seq[t].missing_fields) | {"pore_pressure"}))
    _finalize_many(seq, [1, 2, 3])
    return [(seq, 1), (seq, 2), (seq, 3)]


def correlated_failure_cases() -> list[tuple[list[ObservedTemporalStep], int]]:
    seq = _toe_excavation_sequence()
    for t in [0, 1]:
        seq[t].displacement = "unknown"
        seq[t].crack = "unknown"
        seq[t].monitoring = "unknown"
        seq[t].missing_fields = ("crack", "displacement", "monitoring")
    _finalize_many(seq, [0, 1])
    return [(seq, 0), (seq, 1)]


def drift_conflict_cases() -> list[tuple[list[ObservedTemporalStep], int]]:
    seq = _high_rain_sequence()
    for t in [1, 2, 3]:
        seq[t].pore_pressure = "low"
        seq[t].noisy_fields = tuple(sorted(set(seq[t].noisy_fields) | {"pore_pressure"}))
        seq[t].conflicting_fields = tuple(sorted(set(seq[t].conflicting_fields) | {"pore_pressure"}))
    _finalize_many(seq, [1, 2, 3])
    return [(seq, 1), (seq, 2), (seq, 3)]


def multi_conflict_cases() -> list[tuple[list[ObservedTemporalStep], int, tuple[str, ...], tuple[str, ...]]]:
    seq = _toe_excavation_sequence()
    seq[0].displacement = "low"
    seq[0].crack = "yes"
    seq[0].monitoring = "unknown"
    seq[0].noisy_fields = ("crack", "displacement")
    seq[0].missing_fields = ("monitoring",)
    seq[0].conflicting_fields = ("crack", "displacement")
    finalize_observed_step(seq[0])
    return [
        (
            seq,
            0,
            ("displacement", "crack", "monitoring"),
            ("PorePressure -> Displacement", "Displacement -> Crack"),
        )
    ]


def delayed_response_cases() -> list[tuple[list[ObservedTemporalStep], int]]:
    seq = _high_rain_sequence(length=8)
    for t in [1, 2]:
        seq[t].pore_pressure = "unknown"
        seq[t].missing_fields = ("pore_pressure",)
    seq[3].pore_pressure = "high"
    seq[3].missing_fields = ()
    seq[3].delayed_fields = ("pore_pressure",)
    _finalize_many(seq, [1, 2, 3])
    return [(seq, 1), (seq, 2), (seq, 3)]


def _recall(agent: BaseUncertaintyAgent, cases: list[tuple[list[ObservedTemporalStep], int]]) -> float:
    if not cases:
        return 0.0
    return sum(int(agent.act(seq, t) == "takeover" and seq[t].takeover_required) for seq, t in cases) / len(cases)


def long_missing_takeover_recall(agent: BaseUncertaintyAgent) -> float:
    return _recall(agent, long_missing_cases())


def correlated_failure_recall(agent: BaseUncertaintyAgent) -> float:
    return _recall(agent, correlated_failure_cases())


def drift_conflict_takeover(agent: BaseUncertaintyAgent) -> float:
    return _recall(agent, drift_conflict_cases())


def multi_conflict_audit_score(agent: BaseUncertaintyAgent) -> float:
    cases = multi_conflict_cases()
    total = 0.0
    for seq, t, fields, links in cases:
        if agent.act(seq, t) != "takeover":
            continue
        audit = "\n".join(agent.audit_uncertainty(seq, t))
        field_score = sum(int(field in audit) for field in fields) / len(fields)
        link_score = sum(int(link in audit) for link in links) / len(links)
        total += 0.5 * field_score + 0.5 * link_score
    return total / len(cases)


def delayed_response_safety(agent: BaseUncertaintyAgent) -> float:
    cases = delayed_response_cases()
    safe = 0
    for seq, t in cases:
        action = agent.act(seq, t)
        if seq[t].takeover_required:
            safe += int(action == "takeover")
        else:
            safe += int(action != "takeover")
    return safe / len(cases)


def stress_v32_gated_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["long_missing_takeover_recall"] >= gates["long_missing_takeover_recall"],
        metrics["correlated_failure_recall"] >= gates["correlated_failure_recall"],
        metrics["drift_conflict_takeover"] >= gates["drift_conflict_takeover"],
        metrics["multi_conflict_audit_score"] >= gates["multi_conflict_audit_score"],
        metrics["delayed_response_safety"] >= gates["delayed_response_safety"],
    ]
    if not all(required):
        return 0.0
    return sum(metrics[key] for key in [
        "long_missing_takeover_recall",
        "correlated_failure_recall",
        "drift_conflict_takeover",
        "multi_conflict_audit_score",
        "delayed_response_safety",
    ]) / 5


def evaluate_v32_stress_agent(agent_name: str, config: dict) -> dict[str, float]:
    agent = make_uncertainty_agent(agent_name)
    metrics = {
        "long_missing_takeover_recall": long_missing_takeover_recall(agent),
        "correlated_failure_recall": correlated_failure_recall(agent),
        "drift_conflict_takeover": drift_conflict_takeover(agent),
        "multi_conflict_audit_score": multi_conflict_audit_score(agent),
        "delayed_response_safety": delayed_response_safety(agent),
    }
    metrics["stress_v32_gated_score"] = stress_v32_gated_score(metrics, config["gates"])
    return metrics
