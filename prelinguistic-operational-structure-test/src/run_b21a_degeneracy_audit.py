from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .b21a_degeneracy_audit import run_b21a_degeneracy_audit, write_b21a_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b21a_degeneracy_audit.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b21a_degeneracy_audit(config, seed=args.seed)
    write_b21a_outputs(summary, records)

    print(f"b21a_degeneracy_audit_score={summary['b21a_degeneracy_audit_score']:.3f}")


if __name__ == "__main__":
    main()
