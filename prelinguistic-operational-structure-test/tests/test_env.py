from __future__ import annotations

from src.env import PLOSEnv

from .conftest import small_config


def test_environment_generates_frames() -> None:
    env = PLOSEnv(small_config()["env"], seed=0)
    episode = env.run_episode("occlusion_crossing")
    assert episode["frames"].shape == (20, 64, 64, 3)
    assert episode["past_frames"].shape[0] == 8
    assert episode["future_frames"].shape[0] == 12


def test_inspect_region_returns_metadata() -> None:
    env = PLOSEnv(small_config()["env"], seed=1)
    env.run_episode("budgeted_inspect")
    gt = env.get_ground_truth()
    info = env.inspect_region(gt["critical_inspection_region"])
    assert info["is_critical"] is True
