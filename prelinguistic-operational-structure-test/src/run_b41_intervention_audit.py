from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b41_intervention_degeneracy_audit import run_b41_intervention_audit, write_b41_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b41_intervention_audit.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b41_intervention_audit(config, seed=args.seed)
    write_b41_outputs(summary, records)
    best = max(float(row.get("b41_intervention_audit_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b41_intervention_audit_score={best:.3f}")


if __name__ == "__main__":
    main()

