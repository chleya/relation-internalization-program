from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


ACTIONS = ["noop", "inspect", "improve_drainage", "add_support", "reduce_load"]
PHYSICAL_ACTIONS = ["improve_drainage", "add_support", "reduce_load"]


@dataclass
class Transition:
    before: dict[str, str]
    action: str
    after: dict[str, str]
    reward: float


class ProcessWorld:
    """Small binary process world for active relation learning."""

    def __init__(self, seed: int = 0, mode: str = "base") -> None:
        self.rng = random.Random(seed)
        self.mode = mode
        self.state = self.sample_state()

    def sample_state(self) -> dict[str, str]:
        rainfall = "high" if self.rng.random() < 0.45 else "low"
        load = "high" if self.rng.random() < 0.50 else "low"
        state = {
            "rainfall": rainfall,
            "load": load,
            "drainage": "poor",
            "support": "absent",
            "pore_pressure": "normal",
            "displacement": "normal",
            "risk": "low",
            "warning": "normal",
        }
        return self._evaluate_process(state)

    def reset(self) -> dict[str, str]:
        self.state = self.sample_state()
        return dict(self.state)

    def step(self, action: str) -> Transition:
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        before = dict(self.state)
        after = dict(before)
        self._apply_action(after, action)
        after = self._evaluate_process(after)
        reward = self._reward(after, action)
        self.state = after
        return Transition(before=before, action=action, after=dict(after), reward=reward)

    def _apply_action(self, state: dict[str, str], action: str) -> None:
        if action == "improve_drainage":
            state["drainage"] = "good"
        elif action == "add_support":
            state["support"] = "present"
        elif action == "reduce_load":
            state["load"] = "low"

    def _evaluate_process(self, state: dict[str, str]) -> dict[str, str]:
        out = dict(state)
        if self.mode == "new_support_pressure_link" and out["support"] == "present":
            out["pore_pressure"] = "normal"
        elif out["drainage"] == "good":
            out["pore_pressure"] = "normal"
        elif self.mode == "rule_reversal":
            out["pore_pressure"] = "high" if out["rainfall"] == "low" else "normal"
        elif out["rainfall"] == "high":
            out["pore_pressure"] = "high"
        else:
            out["pore_pressure"] = "normal"

        if out["support"] == "present":
            out["displacement"] = "normal"
        elif out["pore_pressure"] == "high" and out["load"] == "high":
            out["displacement"] = "high"
        else:
            out["displacement"] = "normal"

        out["risk"] = "high" if out["displacement"] == "high" else "low"
        out["warning"] = self._warning(out["risk"])
        return out

    def _warning(self, risk: str) -> str:
        if self.mode == "ood_warning":
            p_warning = 0.2 if risk == "high" else 0.8
        else:
            p_warning = 0.9 if risk == "high" else 0.1
        return "warning" if self.rng.random() < p_warning else "normal"

    def _reward(self, state: dict[str, str], action: str) -> float:
        reward = 1.0 if state["risk"] == "low" else -1.0
        if action in {"improve_drainage", "add_support", "reduce_load"}:
            reward -= 0.1
        elif action == "inspect":
            reward -= 0.05
        return reward

    @staticmethod
    def optimal_actions(state: dict[str, str]) -> set[str]:
        risk_reducing = set()
        for action in PHYSICAL_ACTIONS:
            candidate = dict(state)
            if action == "improve_drainage":
                candidate["drainage"] = "good"
            elif action == "add_support":
                candidate["support"] = "present"
            elif action == "reduce_load":
                candidate["load"] = "low"

            if candidate["drainage"] == "good":
                candidate["pore_pressure"] = "normal"
            elif candidate["rainfall"] == "high":
                candidate["pore_pressure"] = "high"
            else:
                candidate["pore_pressure"] = "normal"

            if candidate["support"] == "present":
                candidate["displacement"] = "normal"
            elif candidate["pore_pressure"] == "high" and candidate["load"] == "high":
                candidate["displacement"] = "high"
            else:
                candidate["displacement"] = "normal"
            candidate["risk"] = "high" if candidate["displacement"] == "high" else "low"
            if state["risk"] == "high" and candidate["risk"] == "low":
                risk_reducing.add(action)

        if risk_reducing:
            return risk_reducing
        if state["risk"] == "low" and state["displacement"] == "normal":
            safe = {"noop", "inspect"}
            if state["pore_pressure"] == "high" and state["drainage"] == "poor":
                safe.add("improve_drainage")
            return safe
        if state["pore_pressure"] == "high" and state["drainage"] == "poor":
            return {"improve_drainage"}
        if state["rainfall"] == "high" and state["drainage"] == "poor":
            return {"improve_drainage"}
        return {"noop", "inspect"}


TRUE_LINKS = {
    ("rainfall=high", "pore_pressure=high"),
    ("drainage=good", "pore_pressure=normal"),
    ("load=high&pore_pressure=high", "displacement=high"),
    ("support=present", "displacement=normal"),
    ("displacement=high", "risk=high"),
    ("displacement=normal", "risk=low"),
    ("action=improve_drainage", "drainage=good"),
    ("action=improve_drainage", "pore_pressure=normal"),
    ("action=add_support", "support=present"),
    ("action=add_support", "displacement=normal"),
    ("action=add_support", "risk=low"),
    ("action=reduce_load", "load=low"),
}

REVERSAL_LINKS = {
    ("rainfall=low", "pore_pressure=high"),
    ("rainfall=high", "pore_pressure=normal"),
    ("drainage=good", "pore_pressure=normal"),
    ("load=high&pore_pressure=high", "displacement=high"),
    ("support=present", "displacement=normal"),
    ("displacement=high", "risk=high"),
    ("displacement=normal", "risk=low"),
    ("action=improve_drainage", "drainage=good"),
    ("action=improve_drainage", "pore_pressure=normal"),
    ("action=add_support", "support=present"),
    ("action=add_support", "displacement=normal"),
    ("action=add_support", "risk=low"),
    ("action=reduce_load", "load=low"),
}


def active_facts(state: dict[str, str], action: str | None = None, ignore_keys: set[str] | None = None) -> set[str]:
    ignored = {"warning"}
    if ignore_keys:
        ignored |= set(ignore_keys)
    singles = sorted(f"{key}={value}" for key, value in state.items() if key not in ignored)
    facts = set(singles)
    for i, left in enumerate(singles):
        for right in singles[i + 1 :]:
            facts.add(f"{left}&{right}")
    if action is not None:
        facts.add(f"action={action}")
    return facts


def outcome_facts(state: dict[str, str], ignore_keys: set[str] | None = None) -> set[str]:
    ignored = {"warning"}
    if ignore_keys:
        ignored |= set(ignore_keys)
    return {f"{key}={value}" for key, value in state.items() if key not in ignored}
