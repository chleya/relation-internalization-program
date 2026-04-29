from pathlib import Path

from src.self_audit import audit_summary


def test_self_audit_names_oracle_limitation():
    summary = Path("results/summary.csv")
    if not summary.exists():
        return
    report = audit_summary(str(summary))
    assert "oracle-style" in report
    assert "not evidence of a deployable engineering AI system" in report
