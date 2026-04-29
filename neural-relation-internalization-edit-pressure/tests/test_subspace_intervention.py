from __future__ import annotations

import torch

from src.intervention import remove_probe_subspace


def test_remove_probe_subspace_preserves_shape() -> None:
    hidden = torch.randn(8, 4)
    probe = torch.nn.Linear(4, 2)
    edited = remove_probe_subspace(hidden, probe)
    assert edited.shape == hidden.shape
    assert torch.norm(edited) <= torch.norm(hidden) + 1e-6
