from __future__ import annotations

from src.case_schema import load_cases
from src.metrics_v5 import replay_consistency, responsibility_traceability
from src.shells import make_shell


def test_no_replay_shell_fails_replay_consistency() -> None:
    case = load_cases("cases/governance_cases.json")[0]
    event = make_shell("no_replay_shell").process(case)
    assert replay_consistency(event) == 0.0


def test_no_responsibility_shell_fails_traceability() -> None:
    case = load_cases("cases/governance_cases.json")[0]
    event = make_shell("no_responsibility_shell").process(case)
    assert responsibility_traceability(event) == 0.0

