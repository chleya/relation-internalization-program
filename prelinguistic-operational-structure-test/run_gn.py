from __future__ import annotations

import json
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from g_line.gn_runner import run_gn

plos_dir = Path(__file__).resolve().parent
with (plos_dir / "configs/g2_compositional.yaml").open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["n_objects"] = 3
config["regions_per_object"] = 5
config["n_train_episodes"] = 24
config["n_test_episodes"] = 12
config["n_ood_episodes"] = 8
config["cs_noise"] = 0.15
config["min_cross_edges"] = 1
config["max_cross_edges"] = 2

print("=== GN: Neural Constructive Generator ===")
metrics = run_gn(config, seed=0)

out_dir = plos_dir / "results" / "gn_neural"
out_dir.mkdir(parents=True, exist_ok=True)

def convert(o):
    if isinstance(o, (bool,)):
        return o
    if hasattr(o, 'item'):
        return o.item()
    return str(o)

(out_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True, default=convert), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
