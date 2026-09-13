from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from g_line.gn_v4_runner import run_gn_continuous

plos_dir = Path(__file__).resolve().parent

print("=== GN v4: Continuous Physics World (ODE) ===")
metrics = run_gn_continuous(n_objects=3, n_train=24, n_test=8, n_ood=8, seed=0)

out_dir = plos_dir / "results" / "gn_continuous"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
