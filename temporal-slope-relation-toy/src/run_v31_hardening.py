from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics_v31_hardening import evaluate_v31_hardening_agent
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
    parser.add_argument("--config", default="configs/v31_hardening.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)

    rows = []
    for agent_name in AGENTS:
        for seed in config["seeds"]:
            rows.append({"agent": agent_name, "seed": seed, **evaluate_v31_hardening_agent(agent_name, config)})

    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/v31_hardening_summary.csv", index=False)
    mean_table = df.groupby("agent").mean(numeric_only=True).round(3).to_string()

    report = f"""# V3.1 Takeover Hardening Report

## 1. Motivation

V3 introduced takeover under missing, noisy, delayed, and conflicting observations. V3.1 attacks a likely false positive: an agent may pass by taking over whenever any uncertainty appears, or by emitting generic audit text.

## 2. Attacks

- irrelevant missing sensor
- benign noise
- takeover overuse
- conflicting downstream evidence
- audit specificity

## 3. Gates

```text
{config["gates"]}
```

## 4. Results

```text
{mean_table}
```

## 5. Interpretation

Non-zero hardening_v31_gated_score requires rejecting irrelevant/benign uncertainty while still taking over for concrete relation-chain conflicts and naming the affected field/link in audit.

## 6. Claim Boundary

Supported:
- toy hardening against generic uncertainty takeover and template-only audit.

Unsupported:
- real engineering takeover policy.
- real monitoring safety model.
"""
    audit = """# V3.1 Self-Audit

## What V3.1 improves

- Penalizes takeover overuse.
- Tests irrelevant missing sensor cases.
- Tests benign noise cases.
- Requires takeover for concrete conflicting evidence.
- Requires audit specificity.

## Remaining weaknesses

- Cases are still hand-constructed toy probes.
- Audit specificity is string-based.
- Real sensor quality and human takeover workflow are not modeled.
- Candidate links remain predefined.
"""
    Path("reports/V3_1_HARDENING_REPORT.md").write_text(report, encoding="utf-8")
    Path("reports/V3_1_SELF_AUDIT.md").write_text(audit, encoding="utf-8")
    print(df.groupby("agent")["hardening_v31_gated_score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()

