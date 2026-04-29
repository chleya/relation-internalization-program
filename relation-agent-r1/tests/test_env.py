from __future__ import annotations

from src.env import ProcessWorld


def test_drainage_reduces_pore_pressure() -> None:
    env = ProcessWorld(seed=1)
    env.state = {
        "rainfall": "high",
        "load": "low",
        "drainage": "poor",
        "support": "absent",
        "pore_pressure": "high",
        "displacement": "normal",
        "risk": "low",
        "warning": "warning",
    }
    transition = env.step("improve_drainage")
    assert transition.after["drainage"] == "good"
    assert transition.after["pore_pressure"] == "normal"


def test_support_reduces_displacement_and_risk() -> None:
    env = ProcessWorld(seed=2)
    env.state = {
        "rainfall": "high",
        "load": "high",
        "drainage": "poor",
        "support": "absent",
        "pore_pressure": "high",
        "displacement": "high",
        "risk": "high",
        "warning": "warning",
    }
    transition = env.step("add_support")
    assert transition.after["support"] == "present"
    assert transition.after["displacement"] == "normal"
    assert transition.after["risk"] == "low"

