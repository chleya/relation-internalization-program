from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def small_config():
    return {
        "seed": 0,
        "env": {
            "frame_size": 64,
            "num_objects": 2,
            "past_frames": 8,
            "future_frames": 12,
            "grid_size": 8,
            "object_radius_min": 3,
            "object_radius_max": 5,
        },
        "data": {"n_train": 4, "n_val": 4, "n_test": 4, "n_ood": 4},
        "runtime": {"max_train_episodes": 4, "max_eval_episodes": 4, "max_ood_episodes": 4},
        "gates": {
            "identity_after_occlusion": 0.85,
            "identity_after_crossing": 0.85,
            "event_boundary_alignment": 0.80,
            "intervention_sensitivity": 0.80,
            "relation_locality": 0.80,
            "noncausal_region_invariance": 0.80,
            "critical_region_selection_accuracy": 0.80,
            "ood_trajectory_generalization": 0.80,
            "slot_causal_drop": 0.20,
            "event_latent_causal_drop": 0.15,
            "relation_edge_causal_drop": 0.20,
            "inspection_map_causal_drop": 0.20,
            "field_causal_drop": 0.20,
            "critical_field_locality": 0.75,
            "noncritical_field_invariance": 0.75,
        },
    }
