from pathlib import Path

from src.b6_4_transfer_generalization.adversarial_review import build_b6_4_adversarial_review
from src.b6_4_transfer_generalization.runner import run_b6_4_transfer, write_b6_4_outputs
from src.b6_4_transfer_generalization.result_review import review_b6_4_results


def test_b64_adversarial_review_flags_shortcut_remaps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary, records, metrics = run_b6_4_transfer({"b6_4": {"episodes_per_condition": 1}}, seed=0)
    write_b6_4_outputs(summary, records, metrics)
    review_b6_4_results()
    review = build_b6_4_adversarial_review()
    assert review["decision"] == "harness_pass_not_strong_transfer_evidence"
    assert review["second_pass_needed"] is True
    assert "visual_remap" in review["summary"]["shortcut_equivalent_remaps"]
    assert Path("results/b6_4_adversarial_review.json").exists()
    assert Path("reports/B6_4_ADVERSARIAL_RESULT_REVIEW.md").exists()
    assert Path("reports/B6_4_SECOND_PASS_HARDENING_PLAN.md").exists()
