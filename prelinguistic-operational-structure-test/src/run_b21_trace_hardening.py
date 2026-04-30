from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b21_trace_attacks import run_b21_trace_hardening, write_b21_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b21_trace_hardening.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b21_trace_hardening(config, seed=args.seed)
    write_b21_outputs(summary, records)

    best = max(float(row.get("b21_trace_hardening_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b21_trace_hardening_score={best:.3f}")


if __name__ == "__main__":
    main()
