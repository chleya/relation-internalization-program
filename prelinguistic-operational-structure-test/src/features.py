from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def frames_to_tensor(frames: np.ndarray) -> np.ndarray:
    return np.asarray(frames, dtype=np.float32)


def region_grid(frame_size: int = 64, grid_size: int = 8) -> list[tuple[int, int, int, int]]:
    cell = frame_size // grid_size
    regions = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            regions.append((gx * cell, gy * cell, (gx + 1) * cell, (gy + 1) * cell))
    return regions


def region_id_to_slice(region_id: int, frame_size: int = 64, grid_size: int = 8) -> tuple[slice, slice]:
    x0, y0, x1, y1 = region_grid(frame_size, grid_size)[int(region_id)]
    return slice(y0, y1), slice(x0, x1)


def point_to_region_id(point: np.ndarray | list[float] | tuple[float, float], frame_size: int = 64, grid_size: int = 8) -> int:
    x, y = float(point[0]), float(point[1])
    cell = frame_size / grid_size
    gx = min(grid_size - 1, max(0, int(x // cell)))
    gy = min(grid_size - 1, max(0, int(y // cell)))
    return gy * grid_size + gx


def rect_to_region_id(rect: tuple[float, float, float, float], frame_size: int = 64, grid_size: int = 8) -> int:
    x0, y0, x1, y1 = rect
    return point_to_region_id(((x0 + x1) / 2.0, (y0 + y1) / 2.0), frame_size, grid_size)


def draw_disk(frame: np.ndarray, center: np.ndarray, radius: float, color: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    x0 = max(0, int(cx - radius - 1))
    x1 = min(w, int(cx + radius + 2))
    y0 = max(0, int(cy - radius - 1))
    y1 = min(h, int(cy + radius + 2))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    frame[y0:y1, x0:x1][mask] = color


def connected_components(mask: np.ndarray, min_size: int = 4) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if visited[y, x] or not mask[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            comp = []
            while queue:
                cy, cx = queue.popleft()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(comp) >= min_size:
                components.append(np.asarray(comp, dtype=np.float32))
    return components


def extract_blob_centers(frame: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 3:
        intensity = image.max(axis=-1)
    else:
        intensity = image
    components = connected_components(intensity > threshold)
    centers = []
    for comp in components:
        yx = comp.mean(axis=0)
        centers.append([float(yx[1]), float(yx[0])])
    if not centers:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def simple_trajectory_extraction(frames: np.ndarray) -> dict[str, Any]:
    centers_by_t = [extract_blob_centers(frame) for frame in frames]
    visible_counts = [len(centers) for centers in centers_by_t]
    last = centers_by_t[-1] if centers_by_t else np.zeros((0, 2), dtype=np.float32)
    prev = next((centers for centers in reversed(centers_by_t[:-1]) if len(centers) == len(last) and len(centers) > 0), last)
    if len(last) and len(prev) == len(last):
        velocity = last - prev
    else:
        velocity = np.zeros_like(last)
    return {
        "centers_by_t": centers_by_t,
        "visible_counts": visible_counts,
        "last_centers": last,
        "velocity": velocity,
    }


def patch_mask(frame_size: int, region_id: int, grid_size: int = 8) -> np.ndarray:
    mask = np.zeros((frame_size, frame_size), dtype=np.float32)
    ys, xs = region_id_to_slice(region_id, frame_size, grid_size)
    mask[ys, xs] = 1.0
    return mask


def region_logits_from_map(value_map: np.ndarray, grid_size: int = 8) -> np.ndarray:
    if value_map.ndim == 3:
        value_map = value_map.mean(axis=-1)
    h, w = value_map.shape
    logits = []
    for ys, xs in [region_id_to_slice(i, h, grid_size) for i in range(grid_size * grid_size)]:
        logits.append(float(value_map[ys, xs].mean()))
    return np.asarray(logits, dtype=np.float32)
