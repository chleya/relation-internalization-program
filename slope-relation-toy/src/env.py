from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RAINFALL = ["low", "high"]
DRAINAGE = ["good", "poor"]
ANCHORING = ["none", "present"]
TOE_EXCAVATION = ["no", "yes"]
MONITORING = ["sparse", "dense"]
WEATHER_LABEL = ["calm", "alarm"]
CONTRACTOR_REPORT = ["normal", "warning"]
ACTIONS = ["monitor", "drain", "anchor", "stop_work"]
RISKS = ["low", "medium", "high"]


@dataclass
class ChainState:
    infiltration: str
    pore_pressure: str
    displacement: str
    crack: str
    risk: str
    optimal_action: str


def evaluate_chain(context: dict[str, str], drainage_effective: bool = True, anchoring_effective: bool = True) -> ChainState:
    infiltration_high = context["rainfall"] == "high"
    pore_high = infiltration_high
    if drainage_effective and context["drainage"] == "good":
        pore_high = False

    displacement_high = pore_high or context["toe_excavation"] == "yes"
    if anchoring_effective and context["anchoring"] == "present":
        displacement_high = False

    crack_yes = displacement_high and context["monitoring"] == "sparse"
    if crack_yes:
        risk = "high"
    elif pore_high or displacement_high:
        risk = "medium"
    else:
        risk = "low"

    if risk == "high":
        action = "stop_work"
    elif pore_high:
        action = "drain"
    elif displacement_high:
        action = "anchor"
    else:
        action = "monitor"

    return ChainState(
        infiltration="high" if infiltration_high else "low",
        pore_pressure="high" if pore_high else "low",
        displacement="high" if displacement_high else "low",
        crack="yes" if crack_yes else "no",
        risk=risk,
        optimal_action=action,
    )


def reward_for(action: str, optimal_action: str, risk: str) -> float:
    if action == optimal_action:
        return 1.0
    if risk == "high" and action == "monitor":
        return -1.0
    if risk == "high":
        return -0.5
    if risk == "medium" and action == "monitor":
        return -0.2
    return 0.0


class SlopeToyWorld:
    def __init__(self, seed: int = 0, mode: str = "base"):
        self.rng = np.random.default_rng(seed)
        self.mode = mode

    def sample_context(self) -> dict[str, str]:
        context = {
            "rainfall": self.rng.choice(RAINFALL).item(),
            "drainage": self.rng.choice(DRAINAGE).item(),
            "anchoring": self.rng.choice(ANCHORING).item(),
            "toe_excavation": self.rng.choice(TOE_EXCAVATION).item(),
            "monitoring": self.rng.choice(MONITORING).item(),
        }
        state = evaluate_chain(context)
        weather, report = self._surface_cues(state.risk)
        context["weather_label"] = weather
        context["contractor_report"] = report
        return context

    def _surface_cues(self, risk: str) -> tuple[str, str]:
        if self.mode == "base":
            preferred = {
                "low": ("calm", "normal"),
                "medium": ("alarm", "normal"),
                "high": ("alarm", "warning"),
            }[risk]
            if self.rng.random() < 0.9:
                return preferred
        if self.mode == "ood":
            return self.rng.choice(WEATHER_LABEL).item(), self.rng.choice(CONTRACTOR_REPORT).item()
        if self.mode == "spurious_attack":
            preferred = {
                "low": ("alarm", "warning"),
                "medium": ("calm", "warning"),
                "high": ("calm", "normal"),
            }[risk]
            if self.rng.random() < 0.9:
                return preferred
        return self.rng.choice(WEATHER_LABEL).item(), self.rng.choice(CONTRACTOR_REPORT).item()

    def step(self, action: str, context: dict[str, str] | None = None) -> dict[str, object]:
        if context is None:
            context = self.sample_context()
        state = evaluate_chain(context)
        reward = reward_for(action, state.optimal_action, state.risk)
        return {"context": context, "state": state, "action": action, "reward": reward}
