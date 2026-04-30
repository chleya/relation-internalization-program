from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b11_flow_attacks import run_b11_flow_hardening, write_b11_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b11_flow_hardening.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    metrics, records = run_b11_flow_hardening(config, seed=args.seed)
    write_b11_outputs(metrics, records)

    print(f"b11_flow_hardening_score={metrics['b11_hardening_score']:.3f}")


if __name__ == "__main__":
    main()
