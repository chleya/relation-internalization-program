from __future__ import annotations

import numpy as np

from src.b2_delayed_metrics import (
    delay_ood_generalization,
    delayed_checkpoint_accuracy,
    delayed_endpoint_shift,
    early_saliency_rejection,
    multi_delay_stability,
)


def test_delayed_checkpoint_accuracy_range():
    assert delayed_checkpoint_accuracy(1, 1) == 1.0
    assert delayed_checkpoint_accuracy(1, 2) == 0.0


def test_early_saliency_rejection_range():
    assert early_saliency_rejection(1, 2) == 1.0
    assert early_saliency_rejection(2, 2) == 0.0


def test_multi_delay_stability_range():
    assert multi_delay_stability(3, 3, {2: 0.2, 4: 1.0}) == 1.0
    assert multi_delay_stability(2, 3, {2: 0.2, 4: 1.0}) == 0.0


def test_delay_ood_generalization_range():
    value = delay_ood_generalization({3: 1.0, 5: 0.0, 7: 0.5})
    assert 0.0 <= value <= 1.0


def test_delayed_endpoint_shift_nonnegative():
    base = np.zeros((2, 64, 64, 3), dtype=np.float32)
    altered = base.copy()
    altered[-1, 20:24, 20:24, :] = 1.0
    assert delayed_endpoint_shift(base, altered) >= 0.0
