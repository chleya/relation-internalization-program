from __future__ import annotations

from src.b31_inspection_degeneracy_audit import compute_inspect_region_overlap


def test_compute_inspect_region_overlap_detects_exact_match():
    rows = []
    for model in ["a", "b", "c"]:
        rows.append(
            {
                "model": model,
                "episode_id": 0,
                "predicted_inspect_region": 7,
                "oracle_best_inspect_region": 7,
                "saliency_region": 3,
            }
        )
    metrics = compute_inspect_region_overlap(rows, {})
    assert metrics["cross_model_inspect_region_match_rate"] == 1.0
    assert metrics["pairwise_region_match_rate"] == 1.0
    assert metrics["gt_region_match_rate"] == 1.0


def test_compute_inspect_region_overlap_accepts_divergence():
    rows = []
    for idx, model in enumerate(["a", "b", "c"]):
        rows.append(
            {
                "model": model,
                "episode_id": 0,
                "predicted_inspect_region": idx,
                "oracle_best_inspect_region": 0,
                "saliency_region": 9,
            }
        )
    metrics = compute_inspect_region_overlap(rows, {})
    assert metrics["cross_model_inspect_region_match_rate"] == 0.0
    assert metrics["pairwise_region_match_rate"] == 0.0
