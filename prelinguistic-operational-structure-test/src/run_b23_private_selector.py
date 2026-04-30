from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b23_selector_validation import run_b23_full_validation, write_b23_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b23_private_selector.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b23_full_validation(config, seed=args.seed)
    write_b23_outputs(summary, records)
    best = max(float(row.get("b23_private_selector_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b23_private_selector_score={best:.3f}")


if __name__ == "__main__":
    main()
