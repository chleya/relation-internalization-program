from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .b6_4_1_transfer_hardening.runner import run_b6_4_1_transfer_hardening, write_b6_4_1_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b6_4_1_transfer_hardening.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records, metrics = run_b6_4_1_transfer_hardening(config, seed=args.seed)
    write_b6_4_1_outputs(summary, records, metrics)
    print(f"hard_transfer_score={metrics['hard_transfer_score']:.3f}")
    print(f"hard_baseline_transfer_gap={metrics['hard_baseline_transfer_gap']:.3f}")
    print(f"shortcut_equivalent_hard_remap_count={metrics['shortcut_equivalent_hard_remap_count']}")


if __name__ == "__main__":
    main()
