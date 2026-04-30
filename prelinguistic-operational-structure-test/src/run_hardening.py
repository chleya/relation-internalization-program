from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .hardening import evaluate_flow_checkpoint_hardening, write_hardening_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    metrics = evaluate_flow_checkpoint_hardening(config, args.seed)
    write_hardening_outputs(metrics)
    print(f"flow_checkpoint_hardening_score={metrics['hardening_score']:.3f}")


if __name__ == "__main__":
    main()
