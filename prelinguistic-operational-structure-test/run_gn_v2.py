from __future__ import annotations

import json
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from g_line.gn_v2_runner import run_gn_contrastive, run_gn_bottleneck

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

print("=== GN v2.1: Contrastive Structure Probing ===")
c_metrics = run_gn_contrastive(config, seed=0)
c_dir = plos_dir / "results" / "gn_contrast"
c_dir.mkdir(parents=True, exist_ok=True)
(c_dir / "metrics.json").write_text(json.dumps(c_metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")

print("\n=== GN v2.2: Bottleneck Compression → 2D Latent ===")
b_metrics = run_gn_bottleneck(config, seed=0)
b_dir = plos_dir / "results" / "gn_bottleneck"
b_dir.mkdir(parents=True, exist_ok=True)
(b_dir / "metrics.json").write_text(json.dumps(b_metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")

all_metrics = {**c_metrics, **b_metrics}
print(f"\n=== SUMMARY ===")
print(f"  contrastive drop  = {c_metrics['gn_contrast_ablation_drop']:.3f}")
print(f"  bottleneck drop   = {b_metrics['gn_bottleneck_ablation_drop']:.3f}")
print("=== DONE ===")
