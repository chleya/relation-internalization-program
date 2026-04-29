from __future__ import annotations

import random
from dataclasses import dataclass


CRITICAL_FIELDS = ["pore_pressure", "anchoring", "displacement", "crack", "risk"]
NONCRITICAL_FIELDS = ["contractor_report", "weather_label", "surface_warning"]
PARTIAL_ACTIONS = ["monitor", "inspect", "improve_drainage", "add_anchoring", "stop_work"]


@dataclass(frozen=True)
class InspectionResult:
    field: str
    observed_value: str
    true_value: str
    accurate: bool


class PartialProcessWorld:
    """Small partial-observation process world for R2.1 hardening."""

    def __init__(self, seed: int = 0, inspection_noise: float = 0.0) -> None:
        self.rng = random.Random(seed)
        self.inspection_noise = inspection_noise

    def sample_state(self) -> dict[str, str]:
        state = {
            "rainfall": "high" if self.rng.random() < 0.45 else "low",
            "drainage": "poor" if self.rng.random() < 0.55 else "good",
            "anchoring": "absent" if self.rng.random() < 0.50 else "present",
            "contractor_report": "delayed" if self.rng.random() < 0.30 else "clear",
            "weather_label": "storm_watch" if self.rng.random() < 0.35 else "clear",
            "surface_warning": "minor" if self.rng.random() < 0.25 else "none",
        }
        return self.evaluate(state)

    def evaluate(self, state: dict[str, str]) -> dict[str, str]:
        out = dict(state)
        if out.get("drainage") == "good":
            out["pore_pressure"] = "normal"
        elif out.get("rainfall") == "high":
            out["pore_pressure"] = "high"
        else:
            out["pore_pressure"] = "normal"

        if out.get("anchoring") == "present":
            out["displacement"] = "normal"
        elif out.get("pore_pressure") == "high":
            out["displacement"] = "high"
        else:
            out["displacement"] = "normal"

        out["crack"] = "open" if out["displacement"] == "high" else "closed"
        out["risk"] = "high" if out["crack"] == "open" else "low"
        return out

    def optimal_action(self, state: dict[str, str]) -> str:
        if state.get("risk") == "high" or state.get("crack") == "open" or state.get("displacement") == "high":
            return "stop_work"
        if state.get("pore_pressure") == "high" and state.get("drainage") == "poor":
            return "improve_drainage"
        if state.get("anchoring") == "absent" and state.get("pore_pressure") == "high":
            return "add_anchoring"
        return "monitor"

    def apply_action(self, state: dict[str, str], action: str) -> dict[str, str]:
        if action not in PARTIAL_ACTIONS:
            raise ValueError(f"unknown action: {action}")
        out = dict(state)
        if action == "improve_drainage":
            out["drainage"] = "good"
        elif action == "add_anchoring":
            out["anchoring"] = "present"
        return self.evaluate(out)

    def inspect(self, true_state: dict[str, str], observed_state: dict[str, str], field: str) -> tuple[dict[str, str], InspectionResult]:
        if field not in true_state:
            raise ValueError(f"cannot inspect unknown field: {field}")
        updated = dict(observed_state)
        true_value = true_state[field]
        observed_value = self._maybe_noisy_value(field, true_value)
        updated[field] = observed_value
        return updated, InspectionResult(
            field=field,
            observed_value=observed_value,
            true_value=true_value,
            accurate=observed_value == true_value,
        )

    def _maybe_noisy_value(self, field: str, value: str) -> str:
        if self.rng.random() >= self.inspection_noise:
            return value
        values = {
            "pore_pressure": ["high", "normal"],
            "anchoring": ["present", "absent"],
            "displacement": ["high", "normal"],
            "crack": ["open", "closed"],
            "risk": ["high", "low"],
            "rainfall": ["high", "low"],
            "drainage": ["good", "poor"],
            "contractor_report": ["clear", "delayed"],
            "weather_label": ["clear", "storm_watch"],
            "surface_warning": ["none", "minor"],
        }.get(field)
        if not values:
            return value
        return values[1] if value == values[0] else values[0]


def mask_fields(state: dict[str, str], fields: list[str]) -> dict[str, str]:
    out = dict(state)
    for field in fields:
        out[field] = "unknown"
    return out


def hard_case_state() -> dict[str, str]:
    return PartialProcessWorld(seed=0).evaluate(
        {
            "rainfall": "high",
            "drainage": "poor",
            "anchoring": "absent",
            "contractor_report": "clear",
            "weather_label": "clear",
            "surface_warning": "none",
        }
    )
