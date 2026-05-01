from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b31_inspection_degeneracy_audit import run_b31_inspection_audit, write_b31_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b31_inspection_audit.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b31_inspection_audit(config, seed=args.seed)
    write_b31_outputs(summary, records)
    best = max(float(row.get("b31_inspection_audit_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b31_inspection_audit_score={best:.3f}")


if __name__ == "__main__":
    main()
