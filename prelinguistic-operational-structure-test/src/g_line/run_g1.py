from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .g1_runner import run_g1, write_g1_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g1_minimal.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, metrics, _ = run_g1(config, seed=args.seed)
    write_g1_outputs(summary, records, metrics)
    print(f"g1_generator_mean_score={metrics['g1_generator_mean_score']:.3f}")
    print(f"g1_ood_score={metrics['g1_ood_score']:.3f}")
    print(f"g1_ood_gain_over_random={metrics['g1_ood_gain_over_random']:.3f}")
    print(f"g1_ood_gain_over_hand_designed={metrics['g1_ood_gain_over_hand_designed']:.3f}")


if __name__ == "__main__":
    main()
