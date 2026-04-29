from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .run_probe import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", default="results/summary.csv")
    parser.add_argument("--train-mode", default="base", choices=["base", "shortcut", "ood", "spurious_attack"])
    parser.add_argument("--train-size", type=int, default=1200)
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        metrics = run(
            seed=seed,
            output=f"results/metrics_{args.train_mode}_seed{seed}.json",
            train_mode=args.train_mode,
            train_size=args.train_size,
        )
        rows.append({"seed": seed, **metrics})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(json.dumps({"output": str(out_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
