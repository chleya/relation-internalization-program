from __future__ import annotations

from src.run_v5_governance import run


def test_v5_run_keeps_only_compliant_shell_gated() -> None:
    rows = run("configs/v5_governance.yaml")
    by_shell = {row["shell"]: row for row in rows}
    assert by_shell["compliant_shell"]["gated_v5_score"] > 0.9
    for shell in ["auto_approve_shell", "no_log_shell", "no_replay_shell", "no_responsibility_shell"]:
        assert by_shell[shell]["gated_v5_score"] == 0.0

