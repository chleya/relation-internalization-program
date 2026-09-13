from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from g_line.g2_runner import run_g2

plos_dir = Path(__file__).resolve().parent
with (plos_dir / "configs/g2_compositional.yaml").open("r", encoding="utf-8") as f:
    import yaml
    config = yaml.safe_load(f)

for name, hardening, out_subdir in [
    ("G2.1 hardening (indirect required)", True, "g2_1_hardening"),
    ("G2.1 baseline re-run (with kill_indirect)", False, "g2_compositional"),
]:
    print(f"\n=== {name} ===")
    summary, records, metrics, artifact = run_g2(config, seed=0, hardening=hardening)

    out_dir = plos_dir / "results" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"  -> saved to {out_dir}")

print("\n=== DONE ===")
