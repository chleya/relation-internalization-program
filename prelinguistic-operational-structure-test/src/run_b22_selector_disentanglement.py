from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b22_selector_disentanglement import run_b22_selector_disentanglement, write_b22_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b22_selector_disentanglement.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b22_selector_disentanglement(config, seed=args.seed)
    write_b22_outputs(summary, records)
    best = max(float(row.get("b22_disentanglement_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b22_disentanglement_score={best:.3f}")


if __name__ == "__main__":
    main()
