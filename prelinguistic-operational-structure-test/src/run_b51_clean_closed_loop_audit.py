from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b51_clean_audit_runner import run_b51_clean_closed_loop_audit, write_b51_clean_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b51_clean_closed_loop_audit.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, leakage = run_b51_clean_closed_loop_audit(config, seed=args.seed)
    write_b51_clean_outputs(summary, records, leakage)
    best = max(float(row.get("b51_clean_closed_loop_audit_score", 0.0)) for row in summary) if summary else 0.0
    leakage_count = sum(int(row.get("leakage_count", 0)) for row in leakage)
    print(f"best_b51_clean_closed_loop_audit_score={best:.3f}")
    print(f"b51_clean_value_leakage_count={leakage_count:.0f}")


if __name__ == "__main__":
    main()
