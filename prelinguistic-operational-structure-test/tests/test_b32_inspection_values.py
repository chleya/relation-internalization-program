from __future__ import annotations

import math

from src.b32_inspection_values import (
    best_region,
    build_family_value_maps,
    compute_combined_inspection_value,
    compute_field_inspection_value,
    compute_recurrent_inspection_value,
    compute_schema_inspection_value,
)
from src.b32_mechanism_inspection_env import make_family_specific_inspection_episode
from tests.conftest import small_config


def test_family_value_maps_have_all_regions_and_distinct_bests():
    config = small_config()
    episode = make_family_specific_inspection_episode(config, 13, "schema_goal")
    maps = build_family_value_maps(episode, config)
    assert len(maps["recurrent_value"]) == 64
    assert len(maps["field_value"]) == 64
    assert len(maps["schema_value"]) == 64
    bests = {
        best_region(maps["recurrent_value"]),
        best_region(maps["field_value"]),
        best_region(maps["schema_value"]),
    }
    assert len(bests) == 3


def test_family_and_combined_values_are_finite():
    config = small_config()
    episode = make_family_specific_inspection_episode(config, 17, "field_goal")
    region = int(episode["ground_truth"]["field_inspect_region"])
    values = [
        compute_recurrent_inspection_value(episode, region, config),
        compute_field_inspection_value(episode, region, config),
        compute_schema_inspection_value(episode, region, config),
        compute_combined_inspection_value(episode, region, [0.0, 1.0, 0.0], config),
    ]
    assert all(math.isfinite(value) for value in values)
