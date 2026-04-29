from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env_temporal import TemporalStep, default_delay_map, generate_temporal_sequence

V3_ACTIONS = ["monitor", "drain", "anchor", "stop_work", "takeover"]
UNKNOWN = "unknown"


@dataclass
class ObservedTemporalStep:
    t: int
    rainfall: str
    drainage: str
    anchoring: str
    toe_excavation: str
    monitoring: str
    surface_warning: str
    weather_label: str
    contractor_report: str
    pore_pressure: str = "low"
    displacement: str = "low"
    crack: str = "no"
    risk: str = "low"
    optimal_action: str = "monitor"
    true_pore_pressure: str = "low"
    true_displacement: str = "low"
    true_crack: str = "no"
    true_risk: str = "low"
    true_optimal_action: str = "monitor"
    observed_automatic_action: str = "monitor"
    takeover_required: bool = False
    takeover_reason: str = ""
    uncertain_links: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    noisy_fields: tuple[str, ...] = ()
    delayed_fields: tuple[str, ...] = ()
    conflicting_fields: tuple[str, ...] = ()

    def as_observed_context(self) -> dict[str, str | int | bool | tuple[str, ...]]:
        return self.__dict__.copy()


def _toggle_value(field: str, value: str) -> str:
    options = {
        "rainfall": ("low", "high"),
        "pore_pressure": ("low", "high"),
        "displacement": ("low", "high"),
        "crack": ("no", "yes"),
        "monitoring": ("sparse", "dense"),
    }
    if field not in options or value == UNKNOWN:
        return value
    a, b = options[field]
    return b if value == a else a


def automatic_action_from_observation(step: ObservedTemporalStep) -> str:
    if step.crack == "yes" or step.risk == "high":
        return "stop_work"
    if step.pore_pressure == "high":
        return "drain"
    if step.displacement == "high":
        return "anchor"
    return "monitor"


def uncertain_links_for_fields(fields: set[str]) -> tuple[str, ...]:
    links: list[str] = []
    if fields & {"rainfall", "pore_pressure"}:
        links.append("Rainfall -> PorePressure")
    if fields & {"drainage", "pore_pressure"}:
        links.append("Drainage -> PorePressureDown")
    if fields & {"pore_pressure", "displacement"}:
        links.append("PorePressure -> Displacement")
    if fields & {"anchoring", "displacement"}:
        links.append("Anchoring -> DisplacementDown")
    if fields & {"displacement", "monitoring", "crack"}:
        links.append("Displacement -> Crack")
    return tuple(dict.fromkeys(links))


def observed_from_true_step(step: TemporalStep) -> ObservedTemporalStep:
    return ObservedTemporalStep(
        t=step.t,
        rainfall=step.rainfall,
        drainage=step.drainage,
        anchoring=step.anchoring,
        toe_excavation=step.toe_excavation,
        monitoring=step.monitoring,
        surface_warning=step.surface_warning,
        weather_label=step.weather_label,
        contractor_report=step.contractor_report,
        pore_pressure=step.pore_pressure,
        displacement=step.displacement,
        crack=step.crack,
        risk=step.risk,
        optimal_action=step.optimal_action,
        true_pore_pressure=step.pore_pressure,
        true_displacement=step.displacement,
        true_crack=step.crack,
        true_risk=step.risk,
        true_optimal_action=step.optimal_action,
    )


def finalize_observed_step(step: ObservedTemporalStep) -> ObservedTemporalStep:
    uncertainty_fields = set(step.missing_fields) | set(step.noisy_fields) | set(step.delayed_fields) | set(step.conflicting_fields)
    observed_auto = automatic_action_from_observation(step)
    step.observed_automatic_action = observed_auto

    critical_missing = bool(set(step.missing_fields) & {"pore_pressure", "displacement", "crack", "monitoring"})
    automatic_wrong = observed_auto != step.true_optimal_action
    critical_true_action = step.true_optimal_action in {"drain", "anchor", "stop_work"}
    step.takeover_required = bool(
        uncertainty_fields
        and (
            automatic_wrong
            or (critical_missing and critical_true_action)
            or len(set(step.missing_fields) & {"pore_pressure", "displacement", "crack"}) >= 2
        )
    )
    step.uncertain_links = uncertain_links_for_fields(uncertainty_fields)
    if step.takeover_required:
        step.optimal_action = "takeover"
        field_text = ", ".join(sorted(uncertainty_fields)) or "unknown field"
        link_text = "; ".join(step.uncertain_links) or "unknown relation"
        step.takeover_reason = f"takeover: uncertain {field_text}; affected links: {link_text}"
    else:
        step.optimal_action = step.true_optimal_action
        step.takeover_reason = ""
    return step


def finalize_observed_sequence(seq: list[ObservedTemporalStep]) -> list[ObservedTemporalStep]:
    for i, step in enumerate(seq):
        finalize_observed_step(step)
        if "pore_pressure" in step.missing_fields and i > 0 and "pore_pressure" in seq[i - 1].missing_fields:
            step.takeover_required = True
            step.optimal_action = "takeover"
            step.uncertain_links = uncertain_links_for_fields({"pore_pressure"})
            step.takeover_reason = "takeover: repeated pore_pressure missing; affected links: Rainfall -> PorePressure; PorePressure -> Displacement"
        if "displacement" in step.missing_fields and i > 0 and "displacement" in seq[i - 1].missing_fields:
            step.takeover_required = True
            step.optimal_action = "takeover"
            step.uncertain_links = uncertain_links_for_fields({"displacement"})
            step.takeover_reason = "takeover: repeated displacement missing; affected links: PorePressure -> Displacement; Displacement -> Crack"
        if "pore_pressure" in step.conflicting_fields and i > 0 and "pore_pressure" in seq[i - 1].conflicting_fields:
            step.takeover_required = True
            step.optimal_action = "takeover"
            step.uncertain_links = uncertain_links_for_fields({"pore_pressure"})
            step.takeover_reason = "takeover: repeated pore_pressure conflict; affected links: Rainfall -> PorePressure; PorePressure -> Displacement"
        if "displacement" in step.conflicting_fields and i > 0 and "displacement" in seq[i - 1].conflicting_fields:
            step.takeover_required = True
            step.optimal_action = "takeover"
            step.uncertain_links = uncertain_links_for_fields({"displacement"})
            step.takeover_reason = "takeover: repeated displacement conflict; affected links: PorePressure -> Displacement; Displacement -> Crack"
    return seq


def apply_observation_uncertainty(
    true_seq: list[TemporalStep],
    rng: np.random.Generator,
    missing_rate: float = 0.18,
    noise_rate: float = 0.12,
    delayed_report_rate: float = 0.15,
    missing_fields: list[str] | None = None,
    noisy_fields: list[str] | None = None,
    delayed_fields: list[str] | None = None,
) -> list[ObservedTemporalStep]:
    missing_fields = missing_fields or ["rainfall", "pore_pressure", "displacement", "crack", "monitoring"]
    noisy_fields = noisy_fields or ["rainfall", "pore_pressure", "displacement", "crack", "monitoring"]
    delayed_fields = delayed_fields or ["pore_pressure", "displacement", "crack"]
    observed = [observed_from_true_step(step) for step in true_seq]

    for t, step in enumerate(observed):
        delayed: list[str] = []
        noisy: list[str] = []
        missing: list[str] = []
        conflicts: list[str] = []

        for field in delayed_fields:
            if t > 0 and rng.random() < delayed_report_rate:
                setattr(step, field, getattr(true_seq[t - 1], field))
                delayed.append(field)

        for field in noisy_fields:
            if rng.random() < noise_rate:
                setattr(step, field, _toggle_value(field, getattr(step, field)))
                noisy.append(field)

        for field in missing_fields:
            if rng.random() < missing_rate:
                setattr(step, field, UNKNOWN)
                missing.append(field)

        true_values = {
            "pore_pressure": step.true_pore_pressure,
            "displacement": step.true_displacement,
            "crack": step.true_crack,
            "risk": step.true_risk,
        }
        for field, true_value in true_values.items():
            observed_value = getattr(step, field)
            if observed_value != UNKNOWN and observed_value != true_value:
                conflicts.append(field)

        step.missing_fields = tuple(sorted(set(missing)))
        step.noisy_fields = tuple(sorted(set(noisy)))
        step.delayed_fields = tuple(sorted(set(delayed)))
        step.conflicting_fields = tuple(sorted(set(conflicts)))
    return finalize_observed_sequence(observed)


def generate_observed_sequence(
    seed: int,
    length: int = 8,
    delay_map: dict[str, int] | None = None,
    missing_rate: float = 0.18,
    noise_rate: float = 0.12,
    delayed_report_rate: float = 0.15,
    mode: str = "v3",
) -> list[ObservedTemporalStep]:
    rng = np.random.default_rng(seed)
    true_seq = generate_temporal_sequence(
        seed=seed,
        length=length,
        delay_map=delay_map or default_delay_map(1),
        shortcut_correlation=0.5,
        mode=mode,
    )
    return apply_observation_uncertainty(
        true_seq,
        rng,
        missing_rate=missing_rate,
        noise_rate=noise_rate,
        delayed_report_rate=delayed_report_rate,
    )
