from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RISKS = ["low", "medium", "high"]
ACTIONS = ["monitor", "drain", "anchor", "stop_work"]


@dataclass
class TemporalStep:
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

    def as_context(self) -> dict[str, str | int]:
        return self.__dict__.copy()


def default_delay_map(rainfall_delay: int = 1) -> dict[str, int]:
    return {
        "Rainfall -> PorePressure": rainfall_delay,
        "PorePressure -> Displacement": 1,
        "Displacement -> Crack": 1,
        "Drainage -> PorePressureDown": 1,
        "Anchoring -> DisplacementDown": 1,
    }


def surface_cues(risk: str, rng: np.random.Generator, mode: str) -> tuple[str, str, str]:
    if mode == "base":
        preferred = {
            "low": ("normal", "clear", "safe"),
            "medium": ("warning", "storm", "safe"),
            "high": ("warning", "storm", "danger"),
        }[risk]
        if rng.random() < 0.9:
            return preferred
    if mode == "spurious_attack":
        preferred = {
            "low": ("warning", "storm", "danger"),
            "medium": ("normal", "clear", "danger"),
            "high": ("normal", "clear", "safe"),
        }[risk]
        if rng.random() < 0.9:
            return preferred
    return (
        rng.choice(["normal", "warning"]).item(),
        rng.choice(["clear", "storm"]).item(),
        rng.choice(["safe", "danger"]).item(),
    )


def assign_false_shortcuts(
    seq: list[TemporalStep],
    rng: np.random.Generator,
    correlation: float,
) -> list[TemporalStep]:
    for step in seq:
        if step.risk == "high":
            correlated = ("warning", "storm", "danger")
            anti = ("normal", "clear", "safe")
        elif step.risk == "medium":
            correlated = ("warning", "storm", "safe")
            anti = ("normal", "clear", "danger")
        else:
            correlated = ("normal", "clear", "safe")
            anti = ("warning", "storm", "danger")
        step.surface_warning, step.weather_label, step.contractor_report = (
            correlated if rng.random() < correlation else anti
        )
    return seq


def generate_sequence(seed: int, length: int = 6, mode: str = "base") -> list[TemporalStep]:
    return generate_temporal_sequence(seed=seed, length=length, mode=mode)


def generate_temporal_sequence(
    seed: int,
    length: int = 8,
    delay_map: dict[str, int] | None = None,
    shortcut_correlation: float = 0.9,
    mode: str = "train",
) -> list[TemporalStep]:
    rng = np.random.default_rng(seed)
    seq: list[TemporalStep] = []
    for t in range(length):
        seq.append(
            TemporalStep(
                t=t,
                rainfall=rng.choice(["low", "high"]).item(),
                drainage=rng.choice(["poor", "good"]).item(),
                anchoring=rng.choice(["none", "present"]).item(),
                toe_excavation=rng.choice(["no", "yes"]).item(),
                monitoring=rng.choice(["sparse", "dense"]).item(),
                surface_warning="normal",
                weather_label="clear",
                contractor_report="safe",
            )
        )
    evaluate_temporal_chain(seq, delay_map=delay_map)
    if mode == "base":
        for step in seq:
            step.surface_warning, step.weather_label, step.contractor_report = surface_cues(step.risk, rng, mode)
    elif mode == "spurious_attack":
        for step in seq:
            step.surface_warning, step.weather_label, step.contractor_report = surface_cues(step.risk, rng, mode)
    else:
        assign_false_shortcuts(seq, rng, shortcut_correlation)
    return seq


def evaluate_temporal_chain(
    seq: list[TemporalStep],
    rainfall_delay: int | None = None,
    delay_map: dict[str, int] | None = None,
) -> list[TemporalStep]:
    if delay_map is None:
        delay_map = default_delay_map(1 if rainfall_delay is None else rainfall_delay)
    d_rain = delay_map["Rainfall -> PorePressure"]
    d_pore = delay_map["PorePressure -> Displacement"]
    d_crack = delay_map["Displacement -> Crack"]
    d_drain = delay_map["Drainage -> PorePressureDown"]
    d_anchor = delay_map["Anchoring -> DisplacementDown"]

    for step in seq:
        step.pore_pressure = "low"
        step.displacement = "low"
        step.crack = "no"
        step.risk = "low"
        step.optimal_action = "monitor"

    for t, step in enumerate(seq):
        rain_t = t - d_rain
        pore_high = rain_t >= 0 and seq[rain_t].rainfall == "high"
        drain_t = t - d_drain
        if drain_t >= 0 and seq[drain_t].drainage == "good":
            pore_high = False
        step.pore_pressure = "high" if pore_high else "low"

        pore_t = t - d_pore
        displacement_high = (pore_t >= 0 and seq[pore_t].pore_pressure == "high") or step.toe_excavation == "yes"
        anchor_t = t - d_anchor
        if anchor_t >= 0 and seq[anchor_t].anchoring == "present":
            displacement_high = False
        step.displacement = "high" if displacement_high else "low"

        crack_t = t - d_crack
        crack_yes = crack_t >= 0 and seq[crack_t].displacement == "high" and seq[crack_t].monitoring == "sparse"
        step.crack = "yes" if crack_yes else "no"

        if step.crack == "yes":
            step.risk = "high"
        elif step.displacement == "high" or step.pore_pressure == "high":
            step.risk = "medium"
        else:
            step.risk = "low"

        if step.risk == "high":
            step.optimal_action = "stop_work"
        elif step.pore_pressure == "high":
            step.optimal_action = "drain"
        elif step.displacement == "high":
            step.optimal_action = "anchor"
        else:
            step.optimal_action = "monitor"
    return seq


def clone_with_rainfall(seq: list[TemporalStep], t: int, rainfall: str) -> list[TemporalStep]:
    copied = [TemporalStep(**step.as_context()) for step in seq]
    copied[t].rainfall = rainfall
    return copied
