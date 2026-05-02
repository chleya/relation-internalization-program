from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .b6_3_structural_necessity.runner import run_b6_3_structural_necessity, write_b6_3_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b6_3_structural_necessity.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, metrics = run_b6_3_structural_necessity(config, seed=args.seed)
    write_b6_3_outputs(summary, records, metrics)
    print(f"best_b6_3_structural_necessity_score={max(float(row['structural_necessity_score']) for row in summary):.3f}")
    print(f"mean_b6_3_structural_necessity_score={metrics['mean_structural_necessity_score']:.3f}")


if __name__ == "__main__":
    main()
