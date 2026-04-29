from __future__ import annotations

from src.case_schema import load_cases
from src.metrics_v5 import score_event
from src.shells import make_shell


def test_compliant_shell_records_replay_and_responsibility() -> None:
    case = load_cases("cases/governance_cases.json")[0]
    event = make_shell("compliant_shell").process(case)
    scores = score_event(case, event)
    assert scores["approval_gate_enforcement"] == 1.0
    assert scores["takeover_routing_quality"] == 1.0
    assert scores["replay_consistency"] == 1.0
    assert scores["responsibility_traceability"] == 1.0


def test_auto_approve_shell_fails_gate() -> None:
    case = load_cases("cases/governance_cases.json")[0]
    event = make_shell("auto_approve_shell").process(case)
    scores = score_event(case, event)
    assert event["auto_approved"] is True
    assert scores["approval_gate_enforcement"] == 0.0

