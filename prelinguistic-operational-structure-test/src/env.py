from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import draw_disk, point_to_region_id, rect_to_region_id


EPISODE_TYPES = ["occlusion_crossing", "collision_bounce", "forcefield", "budgeted_inspect"]


@dataclass
class Ball:
    object_id: int
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    color: np.ndarray


class PLOSEnv:
    def __init__(self, config: dict[str, Any] | None = None, seed: int = 0) -> None:
        self.config = config or {}
        self.frame_size = int(self.config.get("frame_size", 64))
        self.grid_size = int(self.config.get("grid_size", 8))
        self.past_frames = int(self.config.get("past_frames", 8))
        self.future_frames = int(self.config.get("future_frames", 12))
        self.total_frames = self.past_frames + self.future_frames
        self.radius_min = int(self.config.get("object_radius_min", 3))
        self.radius_max = int(self.config.get("object_radius_max", 5))
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.t = 0
        self.episode_type = "occlusion_crossing"
        self.balls: list[Ball] = []
        self.frames: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.occluder = (25.0, 18.0, 39.0, 46.0)
        self.force_field_region = (36.0, 20.0, 48.0, 36.0)
        self.force_vector = np.asarray([1.10, -0.55], dtype=np.float32)
        self.collision_time: int | None = None
        self.collision_pairs: list[tuple[int, int]] = []
        self.forcefield_intervals: list[dict[str, int]] = []
        self.occlusion_intervals: list[dict[str, int]] = []
        self.crossing_intervals: list[dict[str, int]] = []
        self.intervention_effect_metadata: dict[str, Any] = {}

    def reset(self, episode_type: str | None = None) -> np.ndarray:
        self.episode_type = episode_type or EPISODE_TYPES[int(self.rng.integers(0, len(EPISODE_TYPES)))]
        self.t = 0
        self.frames = []
        self.positions = []
        self.velocities = []
        self.collision_time = None
        self.collision_pairs = []
        self.forcefield_intervals = []
        self.occlusion_intervals = []
        self.crossing_intervals = []
        self.intervention_effect_metadata = {}
        self.occluder = self._sample_occluder()
        self.force_field_region = self._sample_force_field()
        self.force_vector = np.asarray([1.10, -0.55], dtype=np.float32)
        self.balls = self._make_balls(self.episode_type)
        frame = self.render_frame()
        self._record(frame)
        return frame

    def step(self) -> np.ndarray:
        self.t += 1
        self._apply_dynamics()
        frame = self.render_frame()
        self._record(frame)
        return frame

    def render_frame(self) -> np.ndarray:
        frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.float32)
        for ball in self.balls:
            if not self._is_occluded(ball.position):
                draw_disk(frame, ball.position, ball.radius, ball.color)
        if self.config.get("render_occluder", True) and self.episode_type in {"occlusion_crossing", "budgeted_inspect"}:
            x0, y0, x1, y1 = [int(v) for v in self.occluder]
            frame[y0:y1, x0:x1, :] = np.maximum(frame[y0:y1, x0:x1, :], 0.08)
        return frame

    def apply_physical_intervention(self, intervention: dict[str, Any]) -> None:
        kind = intervention.get("type")
        if kind == "forcefield_shift":
            dx, dy = intervention.get("delta", (4.0, 0.0))
            x0, y0, x1, y1 = self.force_field_region
            self.force_field_region = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        elif kind == "velocity_kick":
            object_id = int(intervention.get("object_id", 0))
            delta = np.asarray(intervention.get("delta", [0.5, 0.0]), dtype=np.float32)
            self.balls[object_id].velocity += delta
        self.intervention_effect_metadata = {"type": kind, "time": self.t}

    def inspect_region(self, region_id: int) -> dict[str, Any]:
        critical = self._critical_inspection_region()
        return {
            "region_id": int(region_id),
            "is_critical": int(region_id) == int(critical),
            "force_field_region": self.force_field_region if int(region_id) == int(critical) else None,
        }

    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "true_positions": np.asarray(self.positions, dtype=np.float32),
            "true_velocities": np.asarray(self.velocities, dtype=np.float32),
            "true_object_ids": [ball.object_id for ball in self.balls],
            "occlusion_intervals": list(self.occlusion_intervals),
            "crossing_intervals": list(self.crossing_intervals),
            "collision_time": self.collision_time,
            "collision_pairs": list(self.collision_pairs),
            "forcefield_intervals": list(self.forcefield_intervals),
            "force_field_region": self.force_field_region,
            "critical_inspection_region": self._critical_inspection_region(),
            "intervention_effect_metadata": dict(self.intervention_effect_metadata),
            "occluder": self.occluder,
            "episode_type": self.episode_type,
            "event_times": self._event_times(),
            "event_points": self._event_points(),
        }

    def run_episode(self, episode_type: str | None = None) -> dict[str, Any]:
        self.reset(episode_type)
        while len(self.frames) < self.total_frames:
            self.step()
        frames = np.asarray(self.frames[: self.total_frames], dtype=np.float32)
        return {
            "frames": frames,
            "past_frames": frames[: self.past_frames],
            "future_frames": frames[self.past_frames :],
            "ground_truth": self.get_ground_truth(),
        }

    def _make_balls(self, episode_type: str) -> list[Ball]:
        radius = float(self.rng.integers(self.radius_min, self.radius_max + 1))
        if self.config.get("ood_color"):
            c0 = np.asarray([0.90, 0.90, 0.20], dtype=np.float32)
            c1 = np.asarray([0.20, 0.90, 0.45], dtype=np.float32)
        else:
            c0 = np.asarray([0.95, 0.25, 0.25], dtype=np.float32)
            c1 = np.asarray([0.25, 0.65, 1.00], dtype=np.float32)
        jitter = lambda scale=2.5: self.rng.normal(0.0, scale, size=2).astype(np.float32)
        if episode_type == "collision_bounce":
            p0 = np.asarray([17.0, 32.0], dtype=np.float32) + jitter()
            p1 = np.asarray([47.0, 32.0], dtype=np.float32) + jitter()
            v0 = np.asarray([1.55, 0.05], dtype=np.float32)
            v1 = np.asarray([-1.55, -0.05], dtype=np.float32)
        elif episode_type in {"forcefield", "budgeted_inspect"}:
            p0 = np.asarray([10.0, 30.0], dtype=np.float32) + jitter()
            p1 = np.asarray([18.0, 48.0], dtype=np.float32) + jitter()
            v0 = np.asarray([1.45, 0.03], dtype=np.float32)
            v1 = np.asarray([0.95, -0.55], dtype=np.float32)
        else:
            p0 = np.asarray([10.0, 18.0], dtype=np.float32) + jitter()
            p1 = np.asarray([54.0, 46.0], dtype=np.float32) + jitter()
            v0 = np.asarray([1.55, 1.05], dtype=np.float32)
            v1 = np.asarray([-1.55, -1.05], dtype=np.float32)
        speed_scale = float(self.config.get("speed_scale", 1.0))
        if self.config.get("collision_angle_jitter"):
            v0 = v0 + np.asarray([0.0, 0.45], dtype=np.float32)
            v1 = v1 + np.asarray([0.0, -0.45], dtype=np.float32)
        if episode_type in {"forcefield", "budgeted_inspect"}:
            center = p1 if self.config.get("forcefield_ood") else p0
            cx, cy = float(center[0]), float(center[1])
            self.force_field_region = (cx - 6.0, cy - 6.0, cx + 6.0, cy + 6.0)
        return [
            Ball(0, p0, v0 * speed_scale, radius, c0),
            Ball(1, p1, v1 * speed_scale, radius, c1),
        ]

    def _apply_dynamics(self) -> None:
        for ball in self.balls:
            if self.episode_type in {"forcefield", "budgeted_inspect"} and self._inside_rect(ball.position, self.force_field_region):
                self.forcefield_intervals.append({"object_id": ball.object_id, "time": self.t})
                ball.velocity = ball.velocity + self.force_vector
            ball.position = ball.position + ball.velocity
            for axis in (0, 1):
                low = ball.radius
                high = self.frame_size - ball.radius - 1
                if ball.position[axis] <= low or ball.position[axis] >= high:
                    ball.velocity[axis] *= -1
                    ball.position[axis] = np.clip(ball.position[axis], low, high)
                    if self.collision_time is None:
                        self.collision_time = self.t

        if self.episode_type == "collision_bounce":
            b0, b1 = self.balls
            dist = float(np.linalg.norm(b0.position - b1.position))
            if dist <= b0.radius + b1.radius + 0.5:
                b0.velocity, b1.velocity = b1.velocity.copy(), b0.velocity.copy()
                if self.collision_time is None:
                    self.collision_time = self.t
                    self.collision_pairs.append((0, 1))

    def _record(self, frame: np.ndarray) -> None:
        self.frames.append(frame)
        self.positions.append(np.stack([ball.position.copy() for ball in self.balls]))
        self.velocities.append(np.stack([ball.velocity.copy() for ball in self.balls]))
        for ball in self.balls:
            if self._is_occluded(ball.position):
                self.occlusion_intervals.append({"object_id": ball.object_id, "time": self.t})
        if np.linalg.norm(self.balls[0].position - self.balls[1].position) < 10.0:
            self.crossing_intervals.append({"time": self.t})

    def _sample_occluder(self) -> tuple[float, float, float, float]:
        cx = float(self.rng.uniform(27.0, 37.0))
        cy = float(self.rng.uniform(24.0, 40.0))
        return (cx - 7.0, cy - 12.0, cx + 7.0, cy + 12.0)

    def _sample_force_field(self) -> tuple[float, float, float, float]:
        if self.episode_type in {"forcefield", "budgeted_inspect"}:
            cx = float(self.rng.uniform(19.0, 24.0))
            cy = float(self.rng.uniform(33.0, 37.0))
            return (cx - 6.0, cy - 6.0, cx + 6.0, cy + 6.0)
        cx = float(self.rng.uniform(32.0, 48.0))
        cy = float(self.rng.uniform(18.0, 42.0))
        return (cx - 6.0, cy - 6.0, cx + 6.0, cy + 6.0)

    def _is_occluded(self, point: np.ndarray) -> bool:
        if self.episode_type not in {"occlusion_crossing", "budgeted_inspect"}:
            return False
        return self._inside_rect(point, self.occluder)

    @staticmethod
    def _inside_rect(point: np.ndarray, rect: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = rect
        return x0 <= float(point[0]) <= x1 and y0 <= float(point[1]) <= y1

    def _critical_inspection_region(self) -> int:
        if self.episode_type in {"forcefield", "budgeted_inspect"}:
            return rect_to_region_id(self.force_field_region, self.frame_size, self.grid_size)
        if self.collision_time is not None and self.positions:
            return point_to_region_id(self.positions[min(self.collision_time, len(self.positions) - 1)].mean(axis=0), self.frame_size, self.grid_size)
        return rect_to_region_id(self.occluder, self.frame_size, self.grid_size)

    def _event_times(self) -> list[int]:
        if self.episode_type in {"forcefield", "budgeted_inspect"}:
            return sorted(set(int(item["time"]) for item in self.forcefield_intervals[:2] if 0 <= int(item["time"]) < self.total_frames))
        if self.episode_type == "collision_bounce":
            return [int(self.collision_time)] if self.collision_time is not None and 0 <= int(self.collision_time) < self.total_frames else []
        times = []
        times.extend(int(item["time"]) for item in self.crossing_intervals[:2])
        times.extend(int(item["time"]) for item in self.occlusion_intervals[:2])
        return sorted(set(t for t in times if 0 <= t < self.total_frames))

    def _event_points(self) -> list[np.ndarray]:
        points: list[np.ndarray] = []
        if self.episode_type in {"forcefield", "budgeted_inspect"}:
            for item in self.forcefield_intervals[:2]:
                time = min(int(item["time"]), len(self.positions) - 1)
                points.append(self.positions[time][int(item["object_id"])].copy())
            return points
        if self.episode_type == "collision_bounce" and self.collision_time is not None and self.positions:
            time = min(int(self.collision_time), len(self.positions) - 1)
            return [self.positions[time].mean(axis=0).copy()]
        for item in self.crossing_intervals[:2]:
            time = min(int(item["time"]), len(self.positions) - 1)
            points.append(self.positions[time].mean(axis=0).copy())
        for item in self.occlusion_intervals[:2]:
            time = min(int(item["time"]), len(self.positions) - 1)
            points.append(self.positions[time][int(item["object_id"])].copy())
        return points
