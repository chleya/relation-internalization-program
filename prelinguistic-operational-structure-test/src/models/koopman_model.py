from __future__ import annotations

from typing import Any

import numpy as np

from ..features import patch_mask, region_logits_from_map
from ..model_io import assert_clean_model_batch
from .base import BasePLOSModel
from .field_model import _smooth


class KoopmanModel(BasePLOSModel):
    name = "koopman_model"
    structural_family = "low_rank_dynamics"

    def __init__(self, rank: int = 16, latent_size: int = 16) -> None:
        self.rank = rank
        self.latent_size = latent_size
        self.mean: np.ndarray | None = None
        self.basis: np.ndarray | None = None
        self.transition: np.ndarray | None = None

    def fit(self, dataset: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        xs = []
        ys = []
        for batch in dataset:
            past = np.asarray(batch["past_frames"], dtype=np.float32)
            for idx in range(len(past) - 1):
                xs.append(_downsample_gray(past[idx], self.latent_size).reshape(-1))
                ys.append(_downsample_gray(past[idx + 1], self.latent_size).reshape(-1))
        if not xs:
            self._fit_identity()
            return
        x = np.asarray(xs, dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32)
        self.mean = x.mean(axis=0)
        centered = x - self.mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rank = max(1, min(self.rank, vt.shape[0]))
        self.basis = vt[:rank].astype(np.float32)
        zx = centered @ self.basis.T
        zy = (y - self.mean) @ self.basis.T
        ridge = 0.05 * np.eye(rank, dtype=np.float32)
        self.transition = np.linalg.solve(zx.T @ zx + ridge, zx.T @ zy).astype(np.float32)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert_clean_model_batch(batch)
        if self.mean is None or self.basis is None or self.transition is None:
            self._fit_identity()
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        latent = self._encode(past[-1])
        future, latent_rollout = self._rollout(latent, int(batch["future_horizon"]), int(batch["frame_size"]))
        structure = self._structure(past, latent, latent_rollout, int(batch["frame_size"]))
        return {
            "future_frames": future,
            "identity_logits": None,
            "event_logits": structure["event_boundary_map"],
            "inspection_logits": structure["critical_region_logits"],
            "structure": structure,
        }

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        output = self.forward(batch)
        kind = intervention.get("type")
        if kind not in {"event_latent_perturbation", "inspection_map_shuffle", "inspection_topk_zero", "field_patch_mask", "critical_field_zero"}:
            return {"applicable": False}
        future = output["future_frames"].copy()
        region = int(np.argmax(output["inspection_logits"]))
        if kind == "event_latent_perturbation":
            latent = output["structure"]["koopman_latent"].copy()
            if latent.size:
                latent[int(np.argmax(np.abs(latent)))] = 0.0
            future, _ = self._rollout(latent, int(batch["future_horizon"]), int(batch["frame_size"]))
            target = "event_latent_perturbation"
        elif kind == "inspection_map_shuffle":
            future = np.roll(future, shift=8, axis=1)
            target = "inspection"
        else:
            mask = patch_mask(future.shape[1], region)[None, :, :, None]
            future = future * (1.0 - 0.40 * mask)
            target = "field"
        return {"applicable": True, "future_frames": future, "base_future_frames": output["future_frames"], "target": target, "target_region": region}

    def _encode(self, frame: np.ndarray) -> np.ndarray:
        vector = _downsample_gray(frame, self.latent_size).reshape(-1)
        return ((vector - self.mean) @ self.basis.T).astype(np.float32)

    def _decode(self, latent: np.ndarray, frame_size: int) -> np.ndarray:
        vector = np.clip(self.mean + latent @ self.basis, 0.0, 1.0)
        small = vector.reshape(self.latent_size, self.latent_size)
        gray = _upsample_nearest(small, frame_size)
        return np.repeat(gray[:, :, None], 3, axis=-1).astype(np.float32)

    def _rollout(self, latent: np.ndarray, horizon: int, frame_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
        frames = []
        states = []
        current = latent.copy()
        for _ in range(horizon):
            current = current @ self.transition
            states.append(current.copy())
            frames.append(self._decode(current, frame_size))
        return np.asarray(frames, dtype=np.float32), states

    def _structure(self, past: np.ndarray, latent: np.ndarray, latent_rollout: list[np.ndarray], frame_size: int) -> dict[str, Any]:
        last_gray = past[-1].mean(axis=-1)
        recon = self._decode(latent, frame_size).mean(axis=-1)
        residual = np.abs(last_gray - recon)
        if latent_rollout:
            latent_shift = np.abs(latent_rollout[0] - latent)
        else:
            latent_shift = np.abs(latent)
        mode_energy = self._mode_energy_map(latent_shift, frame_size)
        value = _smooth(residual + mode_energy)
        event = (value > max(0.02, float(value.mean() + value.std()))).astype(np.float32)
        return {
            "applicable": True,
            "koopman_latent": latent.astype(np.float32),
            "koopman_transition": self.transition.astype(np.float32),
            "mode_energy_field": mode_energy.astype(np.float32),
            "prediction_residual_field": residual.astype(np.float32),
            "event_boundary_map": event,
            "relation_locality_map": value.astype(np.float32),
            "inspection_value_field": value.astype(np.float32),
            "critical_region_logits": region_logits_from_map(value),
            "intervention_family": "low_rank_dynamics",
        }

    def _mode_energy_map(self, weights: np.ndarray, frame_size: int) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float32)
        basis = np.asarray(self.basis, dtype=np.float32)
        count = min(len(weights), basis.shape[0])
        energy = np.abs(weights[:count] @ basis[:count])
        small = energy.reshape(self.latent_size, self.latent_size)
        if float(small.max()) > 0:
            small = small / float(small.max())
        return _upsample_nearest(small, frame_size)

    def _fit_identity(self) -> None:
        dim = self.latent_size * self.latent_size
        rank = min(self.rank, dim)
        self.mean = np.zeros(dim, dtype=np.float32)
        self.basis = np.eye(dim, dtype=np.float32)[:rank]
        self.transition = np.eye(rank, dtype=np.float32)


def _downsample_gray(frame: np.ndarray, size: int) -> np.ndarray:
    gray = np.asarray(frame, dtype=np.float32).mean(axis=-1)
    h, w = gray.shape
    cell_y = h // size
    cell_x = w // size
    small = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            small[y, x] = float(gray[y * cell_y : (y + 1) * cell_y, x * cell_x : (x + 1) * cell_x].mean())
    return small


def _upsample_nearest(small: np.ndarray, frame_size: int) -> np.ndarray:
    scale = frame_size // small.shape[0]
    return np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)[:frame_size, :frame_size]
