from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .run_extract import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--modes", nargs="+", default=["base", "shortcut"])
    parser.add_argument("--output", default="results/extraction_summary.csv")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for seed in args.seeds:
            metrics = run(
                seed=seed,
                train_mode=mode,
                output=f"results/extracted_{mode}_seed{seed}.json",
            )
            rows.append({key: value for key, value in metrics.items() if key not in {"rules_before_edit", "rules_after_edit"}})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(json.dumps({"output": str(out_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
