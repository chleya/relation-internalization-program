from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .g1_2_runner import run_g1_2, write_g1_2_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g1_2_feature_induction.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, metrics, _ = run_g1_2(config, seed=args.seed)
    write_g1_2_outputs(summary, records, metrics)
    print(f"g1_2_mean_score={metrics['g1_2_mean_score']:.3f}")
    print(f"g1_2_ood_score={metrics['g1_2_ood_score']:.3f}")
    print(f"feedback_feature_drop={metrics['feedback_feature_drop']:.3f}")
    print(f"compression_feature_drop={metrics['compression_feature_drop']:.3f}")
    print(f"gain_over_random_feature_program={metrics['gain_over_random_feature_program']:.3f}")


if __name__ == "__main__":
    main()
