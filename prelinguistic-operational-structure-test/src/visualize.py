from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


FIGURES = [
    "object_persistence.png",
    "event_boundaries.png",
    "relation_locality.png",
    "inspection_regions.png",
    "field_maps.png",
    "structure_intervention_effects.png",
    "plos_candidate_scores.png",
]


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def simple_bar(rows: list[dict[str, str]], key: str, path: str | Path, title: str) -> None:
    models = [row["model"] for row in rows]
    values = [float(row.get(key, 0.0)) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(models, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/overall_summary.csv")
    args = parser.parse_args()
    rows = load_rows(args.summary)
    mapping = {
        "object_persistence.png": ("behavior_score", "Behavior Score"),
        "event_boundaries.png": ("behavior_score", "Event/Behavior Proxy"),
        "relation_locality.png": ("structure_intervention_score", "Structure Intervention Score"),
        "inspection_regions.png": ("behavior_score", "Inspection/Behavior Proxy"),
        "field_maps.png": ("structure_intervention_score", "Field/Structure Proxy"),
        "structure_intervention_effects.png": ("structure_intervention_score", "Intervention Effects"),
        "plos_candidate_scores.png": ("plos_candidate_score", "PLOS Candidate Score"),
    }
    for fig in FIGURES:
        key, title = mapping[fig]
        simple_bar(rows, key, Path("figures") / fig, title)


if __name__ == "__main__":
    main()
