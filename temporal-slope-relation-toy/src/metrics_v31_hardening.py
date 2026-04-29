from __future__ import annotations

from .agents_uncertainty import BaseUncertaintyAgent, make_uncertainty_agent
from .env_temporal import TemporalStep, evaluate_temporal_chain
from .env_uncertainty import ObservedTemporalStep, apply_observation_uncertainty, finalize_observed_step


def _clean_observed(seq: list[TemporalStep]) -> list[ObservedTemporalStep]:
    import numpy as np

    return apply_observation_uncertainty(seq, np.random.default_rng(0), 0.0, 0.0, 0.0)


def _low_risk_sequence() -> list[ObservedTemporalStep]:
    seq = [
        TemporalStep(0, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(2, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    return _clean_observed(seq)


def _rain_to_pore_missing_case() -> list[ObservedTemporalStep]:
    seq = [
        TemporalStep(0, "high", "poor", "none", "no", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    observed = _clean_observed(seq)
    observed[1].pore_pressure = "unknown"
    observed[1].missing_fields = ("pore_pressure",)
    finalize_observed_step(observed[1])
    return observed


def _displacement_conflict_case() -> list[ObservedTemporalStep]:
    seq = [
        TemporalStep(0, "low", "poor", "none", "yes", "dense", "normal", "clear", "safe"),
        TemporalStep(1, "low", "poor", "none", "no", "dense", "normal", "clear", "safe"),
    ]
    evaluate_temporal_chain(seq, rainfall_delay=1)
    observed = _clean_observed(seq)
    observed[0].displacement = "low"
    observed[0].noisy_fields = ("displacement",)
    observed[0].conflicting_fields = ("displacement",)
    finalize_observed_step(observed[0])
    return observed


def _irrelevant_missing_cases() -> list[tuple[list[ObservedTemporalStep], int]]:
    cases = []
    observed = _low_risk_sequence()
    observed[1].surface_warning = "unknown"
    observed[1].missing_fields = ("surface_warning",)
    finalize_observed_step(observed[1])
    cases.append((observed, 1))

    observed = _low_risk_sequence()
    observed[1].contractor_report = "unknown"
    observed[1].missing_fields = ("contractor_report",)
    finalize_observed_step(observed[1])
    cases.append((observed, 1))
    return cases


def _benign_noise_cases() -> list[tuple[list[ObservedTemporalStep], int]]:
    cases = []
    observed = _low_risk_sequence()
    observed[1].monitoring = "sparse"
    observed[1].noisy_fields = ("monitoring",)
    finalize_observed_step(observed[1])
    cases.append((observed, 1))

    observed = _low_risk_sequence()
    observed[1].weather_label = "storm"
    observed[1].noisy_fields = ("weather_label",)
    finalize_observed_step(observed[1])
    cases.append((observed, 1))
    return cases


def _safe_uncertainty_cases() -> list[tuple[list[ObservedTemporalStep], int]]:
    cases = []
    observed = _low_risk_sequence()
    observed[1].pore_pressure = "unknown"
    observed[1].missing_fields = ("pore_pressure",)
    finalize_observed_step(observed[1])
    cases.append((observed, 1))

    observed = _low_risk_sequence()
    observed[1].crack = "unknown"
    observed[1].missing_fields = ("crack",)
    finalize_observed_step(observed[1])
    cases.append((observed, 1))

    cases.extend(_irrelevant_missing_cases())
    cases.extend(_benign_noise_cases())
    return cases


def _takeover_required_cases() -> list[tuple[list[ObservedTemporalStep], int, str, str]]:
    return [
        (_rain_to_pore_missing_case(), 1, "pore_pressure", "Rainfall -> PorePressure"),
        (_displacement_conflict_case(), 0, "displacement", "PorePressure -> Displacement"),
    ]


def irrelevant_missing_rejection(agent: BaseUncertaintyAgent) -> float:
    cases = _irrelevant_missing_cases()
    return sum(int(agent.act(seq, t) != "takeover" and not seq[t].takeover_required) for seq, t in cases) / len(cases)


def benign_noise_rejection(agent: BaseUncertaintyAgent) -> float:
    cases = _benign_noise_cases()
    return sum(int(agent.act(seq, t) != "takeover" and not seq[t].takeover_required) for seq, t in cases) / len(cases)


def takeover_overuse_control(agent: BaseUncertaintyAgent) -> float:
    cases = _safe_uncertainty_cases()
    return sum(int(agent.act(seq, t) != "takeover" and not seq[t].takeover_required) for seq, t in cases) / len(cases)


def conflicting_evidence_takeover(agent: BaseUncertaintyAgent) -> float:
    cases = _takeover_required_cases()
    return sum(int(agent.act(seq, t) == "takeover" and seq[t].takeover_required) for seq, t, _, _ in cases) / len(cases)


def audit_specificity(agent: BaseUncertaintyAgent) -> float:
    cases = _takeover_required_cases()
    total = 0.0
    for seq, t, field, link in cases:
        audit = "\n".join(agent.audit_uncertainty(seq, t))
        has_takeover = "takeover" in audit.lower()
        has_field = field in audit
        has_link = link in audit
        total += (int(has_takeover) + int(has_field) + int(has_link)) / 3
    return total / len(cases)


def hardening_v31_gated_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["irrelevant_missing_rejection"] >= gates["irrelevant_missing_rejection"],
        metrics["benign_noise_rejection"] >= gates["benign_noise_rejection"],
        metrics["takeover_overuse_control"] >= gates["takeover_overuse_control"],
        metrics["conflicting_evidence_takeover"] >= gates["conflicting_evidence_takeover"],
        metrics["audit_specificity"] >= gates["audit_specificity"],
    ]
    if not all(required):
        return 0.0
    return (
        0.20 * metrics["irrelevant_missing_rejection"]
        + 0.20 * metrics["benign_noise_rejection"]
        + 0.20 * metrics["takeover_overuse_control"]
        + 0.20 * metrics["conflicting_evidence_takeover"]
        + 0.20 * metrics["audit_specificity"]
    )


def evaluate_v31_hardening_agent(agent_name: str, config: dict) -> dict[str, float]:
    agent = make_uncertainty_agent(agent_name)
    metrics = {
        "irrelevant_missing_rejection": irrelevant_missing_rejection(agent),
        "benign_noise_rejection": benign_noise_rejection(agent),
        "takeover_overuse_control": takeover_overuse_control(agent),
        "conflicting_evidence_takeover": conflicting_evidence_takeover(agent),
        "audit_specificity": audit_specificity(agent),
    }
    metrics["hardening_v31_gated_score"] = hardening_v31_gated_score(metrics, config["gates"])
    return metrics

