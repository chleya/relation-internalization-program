from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b6_hardening.hardening_runner import run_b6_1_hardening, write_b6_1_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b6_1_hardening.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, metrics = run_b6_1_hardening(config, seed=args.seed)
    write_b6_1_outputs(summary, records, metrics)
    hardening = [float(row["risk_constrained_score"]) for row in summary if row.get("policy_name") == "hardening_policy"]
    best = max(hardening) if hardening else 0.0
    mean_score = sum(hardening) / max(1, len(hardening))
    print(f"best_b6_1_hardening_score={best:.3f}")
    print(f"mean_b6_1_hardening_score={mean_score:.3f}")


if __name__ == "__main__":
    main()

