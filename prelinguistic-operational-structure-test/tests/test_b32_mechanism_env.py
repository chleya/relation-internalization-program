from __future__ import annotations

from src.b32_mechanism_inspection_env import make_family_specific_inspection_episode, make_mechanism_disagreement_episode
from tests.conftest import small_config


def test_family_specific_episode_has_distinct_targets():
    episode = make_family_specific_inspection_episode(small_config(), 7, "recurrent_goal")
    gt = episode["ground_truth"]
    regions = {
        int(gt["recurrent_inspect_region"]),
        int(gt["field_inspect_region"]),
        int(gt["schema_inspect_region"]),
    }
    assert len(regions) == 3
    assert "saliency_region" in gt
    assert "short_horizon_region" in gt
    assert gt["episode_type"] == "b32_family_specific_inspection"


def test_mechanism_disagreement_episode_marks_episode_type():
    episode = make_mechanism_disagreement_episode(small_config(), 11)
    assert episode["ground_truth"]["episode_type"] == "b32_mechanism_disagreement"
