from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics_v32_stress import evaluate_v32_stress_agent
from .utils import load_yaml


AGENTS = [
    "surface_temporal",
    "structural_memory_temporal",
    "instant_relation_chain",
    "delayed_relation_chain",
    "learned_delayed_links",
    "uncertainty_aware_delayed_links",
    "overcautious_takeover",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v32_stress.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)

    rows = []
    for agent_name in AGENTS:
        for seed in config["seeds"]:
            rows.append({"agent": agent_name, "seed": seed, **evaluate_v32_stress_agent(agent_name, config)})

    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/v32_stress_summary.csv", index=False)
    mean_table = df.groupby("agent").mean(numeric_only=True).round(3).to_string()

    report = f"""# V3.2 Partial-Observability Stress Report

## 1. Motivation

V3.1 hardened takeover against generic overcaution. V3.2 stress-tests harder partial-observability cases: long missing spans, correlated sensor failure, drift-like conflicts, multi-conflict audit, and delayed response safety.

## 2. Stress Tests

- long missing sensor span
- correlated sensor failure
- drift conflict
- multi-conflict audit
- delayed response safety

## 3. Gates

```text
{config["gates"]}
```

## 4. Results

```text
{mean_table}
```

## 5. Interpretation

Non-zero stress_v32_gated_score requires takeover behavior to remain safe when observation uncertainty lasts across time or affects multiple related sensors.

## 6. Claim Boundary

Supported:
- toy stress testing for partial-observability relation-chain takeover.

Unsupported:
- real-world monitoring reliability.
- validated engineering takeover policy.
"""
    audit = """# V3.2 Self-Audit

## What V3.2 improves

- Tests long missing spans.
- Tests correlated sensor failures.
- Tests drift-like conflicts.
- Tests multi-conflict audit specificity.
- Tests delayed response safety.

## Remaining weaknesses

- Stress cases are still hand-constructed.
- No real sensor time series.
- No probabilistic calibration of uncertainty.
- No human response model.
- Candidate relation links remain predefined.
"""
    Path("reports/V3_2_STRESS_REPORT.md").write_text(report, encoding="utf-8")
    Path("reports/V3_2_SELF_AUDIT.md").write_text(audit, encoding="utf-8")
    print(df.groupby("agent")["stress_v32_gated_score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()

