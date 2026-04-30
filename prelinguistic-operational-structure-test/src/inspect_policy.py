from __future__ import annotations

import numpy as np


def select_region_from_logits(logits: np.ndarray | None) -> int:
    if logits is None:
        return 0
    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0
    return int(np.argmax(values))


def inspection_value_gain(selected_region: int, critical_region: int) -> float:
    return 1.0 if int(selected_region) == int(critical_region) else 0.0
