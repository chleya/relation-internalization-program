from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .features import draw_disk, point_to_region_id, region_id_to_slice


BALL_COLORS = [
    np.asarray([0.95, 0.25, 0.25], dtype=np.float32),
    np.asarray([0.25, 0.65, 1.00], dtype=np.float32),
]
TRACE_COLOR = np.asarray([0.14, 0.12, 0.04], dtype=np.float32)
DECOY_COLOR = np.asarray([0.86, 0.86, 0.86], dtype=np.float32)


def make_delayed_checkpoint_episode(
    config: dict[str, Any],
    seed: int,
    delay: int,
    saliency_decoy: bool = True,
) -> dict[str, Any]:
    """
    Generate a delayed checkpoint episode.

    The low-intensity trace is visible in W as continuous dynamics, but it is
    below the object/saliency channel used by the short-horizon checkpoint
    baseline. The model must bind that trace to future endpoint consequences.
    """

    base = resolve_base_config(config)
    env = env_config(base, config)
    rng = np.random.default_rng(seed)
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    past_frames = int(env.get("past_frames", 8))
    future_frames = int(env.get("future_frames", 12))
    total_frames = past_frames + future_frames

    p0 = np.asarray([8.0, float(rng.uniform(22.0, 38.0))], dtype=np.float32)
    v0 = np.asarray([2.25, float(rng.uniform(-0.16, 0.16))], dtype=np.float32)
    p1 = np.asarray([52.0, float(rng.uniform(18.0, 46.0))], dtype=np.float32)
    v1 = np.asarray([-0.35, float(rng.uniform(-0.10, 0.10))], dtype=np.float32)
    causal_time = min(past_frames + int(delay), total_frames - 2)
    trace_point = p0 + v0 * float(causal_time)
    trace_point = np.clip(trace_point, 6.0, frame_size - 7.0)
    true_region = int(point_to_region_id(trace_point, frame_size, grid_size))
    early_region = choose_far_region(true_region, grid_size, seed + 1)
    force_vector = delayed_force_vector(seed + 2)
    episode = render_delayed_episode(
        frame_size=frame_size,
        grid_size=grid_size,
        past_frames=past_frames,
        future_frames=future_frames,
        p0=p0,
        v0=v0,
        p1=p1,
        v1=v1,
        true_region=true_region,
        causal_time=causal_time,
        trace_regions={true_region: 1.0},
        early_saliency_region=early_region if saliency_decoy else None,
        force_vector=force_vector,
    )
    episode["ground_truth"].update(
        {
            "episode_type": "b2_delayed_checkpoint",
            "delay": int(delay),
            "true_delayed_checkpoint_region": int(true_region),
            "early_saliency_region": int(early_region),
            "critical_inspection_region": int(true_region),
            "delayed_causal_time": int(causal_time),
            "delayed_endpoint_reference": episode["ground_truth"]["true_positions"][-1, 0].tolist(),
        }
    )
    return episode


def make_multi_delay_checkpoint_episode(config: dict[str, Any], seed: int, delays: list[int]) -> dict[str, Any]:
    """
    Generate an episode with multiple candidate delayed checkpoints.
    """

    base = resolve_base_config(config)
    env = env_config(base, config)
    rng = np.random.default_rng(seed)
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    past_frames = int(env.get("past_frames", 8))
    future_frames = int(env.get("future_frames", 12))
    total_frames = past_frames + future_frames

    p0 = np.asarray([7.5, float(rng.uniform(21.0, 39.0))], dtype=np.float32)
    v0 = np.asarray([2.12, float(rng.uniform(-0.18, 0.18))], dtype=np.float32)
    p1 = np.asarray([53.0, float(rng.uniform(18.0, 46.0))], dtype=np.float32)
    v1 = np.asarray([-0.30, float(rng.uniform(-0.12, 0.12))], dtype=np.float32)
    values = {}
    candidate_regions = []
    for idx, delay in enumerate(delays):
        causal_time = min(past_frames + int(delay), total_frames - 2)
        point = np.clip(p0 + v0 * float(causal_time), 6.0, frame_size - 7.0)
        region = int(point_to_region_id(point, frame_size, grid_size))
        candidate_regions.append(region)
        values[int(delay)] = 0.45 + 0.08 * idx
    best_delay = int(delays[int(rng.integers(0, len(delays)))])
    values[best_delay] = 1.0
    best_index = delays.index(best_delay)
    best_region = int(candidate_regions[best_index])
    early_region = choose_far_region(best_region, grid_size, seed + 5)
    trace_regions = {int(region): float(values[int(delay)]) for region, delay in zip(candidate_regions, delays)}
    episode = render_delayed_episode(
        frame_size=frame_size,
        grid_size=grid_size,
        past_frames=past_frames,
        future_frames=future_frames,
        p0=p0,
        v0=v0,
        p1=p1,
        v1=v1,
        true_region=best_region,
        causal_time=min(past_frames + best_delay, total_frames - 2),
        trace_regions=trace_regions,
        early_saliency_region=early_region,
        force_vector=delayed_force_vector(seed + 6),
    )
    episode["ground_truth"].update(
        {
            "episode_type": "b2_multi_delay_checkpoint",
            "candidate_delays": [int(delay) for delay in delays],
            "candidate_regions": [int(region) for region in candidate_regions],
            "best_delay": int(best_delay),
            "best_region": int(best_region),
            "delay_values": {int(delay): float(value) for delay, value in values.items()},
            "delay_region_values": {int(region): float(values[int(delay)]) for region, delay in zip(candidate_regions, delays)},
            "true_delayed_checkpoint_region": int(best_region),
            "early_saliency_region": int(early_region),
            "critical_inspection_region": int(best_region),
            "delayed_causal_time": int(min(past_frames + best_delay, total_frames - 2)),
        }
    )
    return episode


def make_delay_ood_episode(config: dict[str, Any], seed: int, heldout_delay: int) -> dict[str, Any]:
    """
    Generate delayed checkpoint with a delay not seen in training.
    """

    episode = make_delayed_checkpoint_episode(config, seed, heldout_delay, saliency_decoy=True)
    episode["ground_truth"]["episode_type"] = "b2_delay_ood"
    episode["ground_truth"]["ood_type"] = "heldout_delay"
    episode["ground_truth"]["heldout_delay"] = int(heldout_delay)
    return episode


def make_b2_datasets(config: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    b2 = b2_config(config)
    delays = [int(value) for value in b2.get("delays", [2, 4, 6])]
    heldout = [int(value) for value in b2.get("heldout_delays", [3, 5, 7])]
    n_train = int(b2.get("n_train", 32))
    n_test = int(b2.get("n_test", 32))
    n_ood = int(b2.get("n_ood", 32))
    return {
        "train": [make_delayed_checkpoint_episode(config, seed + idx * 11, delays[idx % len(delays)]) for idx in range(n_train)],
        "delayed": [make_delayed_checkpoint_episode(config, seed + 10000 + idx * 13, delays[idx % len(delays)]) for idx in range(n_test)],
        "multi": [make_multi_delay_checkpoint_episode(config, seed + 20000 + idx * 17, delays) for idx in range(n_test)],
        "ood": [make_delay_ood_episode(config, seed + 30000 + idx * 19, heldout[idx % len(heldout)]) for idx in range(n_ood)],
    }


def render_delayed_episode(
    frame_size: int,
    grid_size: int,
    past_frames: int,
    future_frames: int,
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    true_region: int,
    causal_time: int,
    trace_regions: dict[int, float],
    early_saliency_region: int | None,
    force_vector: np.ndarray,
) -> dict[str, Any]:
    total_frames = past_frames + future_frames
    positions = []
    velocities = []
    frames = []
    current_p0 = p0.copy()
    current_p1 = p1.copy()
    current_v0 = v0.copy()
    current_v1 = v1.copy()
    no_force_p0 = p0.copy()
    no_force_v0 = v0.copy()
    no_force_endpoint = None
    force_applied = False

    for t in range(total_frames):
        frame = np.zeros((frame_size, frame_size, 3), dtype=np.float32)
        if t < past_frames:
            for region, value in trace_regions.items():
                draw_trace(frame, region, frame_size, grid_size, value)
            if early_saliency_region is not None:
                draw_saliency_decoy(frame, early_saliency_region, frame_size, grid_size, t, past_frames)
        draw_disk(frame, current_p0, 4.0, BALL_COLORS[0])
        draw_disk(frame, current_p1, 4.0, BALL_COLORS[1])
        frames.append(frame)
        positions.append(np.stack([current_p0.copy(), current_p1.copy()]))
        velocities.append(np.stack([current_v0.copy(), current_v1.copy()]))

        if t == causal_time and int(point_to_region_id(current_p0, frame_size, grid_size)) == int(true_region):
            current_v0 = current_v0 + force_vector
            force_applied = True
        current_p0, current_v0 = step_ball(current_p0, current_v0, frame_size)
        current_p1, current_v1 = step_ball(current_p1, current_v1, frame_size)

        no_force_p0, no_force_v0 = step_ball(no_force_p0, no_force_v0, frame_size)
        no_force_endpoint = no_force_p0.copy()

    frames_array = np.asarray(frames, dtype=np.float32)
    positions_array = np.asarray(positions, dtype=np.float32)
    velocities_array = np.asarray(velocities, dtype=np.float32)
    return {
        "frames": frames_array,
        "past_frames": frames_array[:past_frames],
        "future_frames": frames_array[past_frames:],
        "ground_truth": {
            "true_positions": positions_array,
            "true_velocities": velocities_array,
            "true_object_ids": [0, 1],
            "occlusion_intervals": [],
            "crossing_intervals": [],
            "collision_time": None,
            "collision_pairs": [],
            "forcefield_intervals": [{"object_id": 0, "time": int(causal_time)}] if force_applied else [],
            "force_field_region": region_bounds(true_region, frame_size, grid_size),
            "critical_inspection_region": int(true_region),
            "intervention_effect_metadata": {"type": "delayed_force", "time": int(causal_time)},
            "occluder": None,
            "episode_type": "b2_delayed_checkpoint",
            "event_times": [int(causal_time)],
            "event_points": [positions_array[min(causal_time, len(positions_array) - 1), 0].copy()],
            "delayed_no_force_endpoint": no_force_endpoint.tolist() if no_force_endpoint is not None else [],
        },
    }


def draw_trace(frame: np.ndarray, region: int, frame_size: int, grid_size: int, value: float) -> None:
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    center = np.asarray([(xs.start + xs.stop - 1) / 2.0, (ys.start + ys.stop - 1) / 2.0], dtype=np.float32)
    color = TRACE_COLOR * float(np.clip(value, 0.35, 1.0))
    draw_disk(frame, center, 2.0, color)


def draw_saliency_decoy(frame: np.ndarray, region: int, frame_size: int, grid_size: int, t: int, past_frames: int) -> None:
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    center = np.asarray([(xs.start + xs.stop - 1) / 2.0, (ys.start + ys.stop - 1) / 2.0], dtype=np.float32)
    offset = np.asarray([(float(t) - (past_frames - 1) / 2.0) * 0.45, 0.0], dtype=np.float32)
    draw_disk(frame, center + offset, 3.0, DECOY_COLOR)


def step_ball(position: np.ndarray, velocity: np.ndarray, frame_size: int) -> tuple[np.ndarray, np.ndarray]:
    position = position + velocity
    velocity = velocity.copy()
    for axis in (0, 1):
        if position[axis] < 4.0 or position[axis] > frame_size - 5.0:
            velocity[axis] *= -1.0
            position[axis] = np.clip(position[axis], 4.0, frame_size - 5.0)
    return position.astype(np.float32), velocity.astype(np.float32)


def delayed_force_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = -1.0 if rng.random() < 0.5 else 1.0
    return np.asarray([0.15, direction * 1.55], dtype=np.float32)


def choose_far_region(region: int, grid_size: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    gy, gx = divmod(int(region), grid_size)
    candidates = []
    for candidate in range(grid_size * grid_size):
        cy, cx = divmod(candidate, grid_size)
        candidates.append((abs(cy - gy) + abs(cx - gx), candidate))
    farthest = max(distance for distance, _ in candidates)
    far = [candidate for distance, candidate in candidates if distance == farthest]
    return int(far[int(rng.integers(0, len(far)))])


def region_bounds(region: int, frame_size: int, grid_size: int) -> tuple[float, float, float, float]:
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    return (float(xs.start), float(ys.start), float(xs.stop), float(ys.stop))


def resolve_base_config(config: dict[str, Any]) -> dict[str, Any]:
    if "env" in config and "gates" in config:
        return config
    base_path = Path(str(config.get("base_config", "configs/sweep.yaml")))
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    with base_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b2_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get(
        "b2",
        {
            "n_train": 32,
            "n_test": 32,
            "n_ood": 32,
            "delays": [2, 4, 6],
            "heldout_delays": [3, 5, 7],
        },
    )


def env_config(base_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_config.get("env", {}))
    b2 = b2_config(config)
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b2:
            merged[key] = b2[key]
    return merged
