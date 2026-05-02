from pathlib import Path

from src.b6_4_1_transfer_hardening.adversarial_review import build_b6_4_1_adversarial_review
from src.b6_4_1_transfer_hardening.result_review import review_b6_4_1_results
from src.b6_4_1_transfer_hardening.runner import run_b6_4_1_transfer_hardening, write_b6_4_1_outputs


def test_b641_adversarial_review_records_combined_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary, records, metrics = run_b6_4_1_transfer_hardening({"b6_4_1": {"episodes_per_condition": 2}}, seed=0)
    write_b6_4_1_outputs(summary, records, metrics)
    review_b6_4_1_results()
    review = build_b6_4_1_adversarial_review()
    assert review["decision"] == "submit_ready_as_diagnostic_branch"
    assert review["hidden_cue_audit"]["hidden_answer_cue_count"] == 0
    assert "combined_remap_hard" in review["combined_remap_hard_conclusion"]
    assert Path("results/b6_4_1_adversarial_review.json").exists()
    assert Path("reports/B6_4_1_ADVERSARIAL_RESULT_REVIEW.md").exists()
