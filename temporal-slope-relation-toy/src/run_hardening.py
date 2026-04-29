from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agents_temporal import make_agent
from .metrics_temporal import evaluate_hardening_agent
from .utils import load_yaml


AGENTS = [
    "surface_temporal",
    "structural_memory_temporal",
    "instant_relation_chain",
    "delayed_relation_chain",
    "learned_delayed_links",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hardening.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)

    rows = []
    for agent_name in AGENTS:
        for seed in config["seeds"]:
            metrics = evaluate_hardening_agent(make_agent(agent_name), seed, config)
            rows.append({"agent": agent_name, "seed": seed, **metrics})

    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/hardening_summary.csv", index=False)
    mean_table = df.groupby("agent").mean(numeric_only=True).round(3).to_string()

    report = f"""# V2.1 Reviewer Hardening Report

## 1. Motivation

V2 needed hardening because fixed delay templates and limited predefined candidates could create false positives.

## 2. New Attacks

- variable delay
- false temporal shortcut
- multi-link delay edit
- temporal audit consistency
- anti-template generalization

## 3. Agents

{", ".join(AGENTS)}

## 4. Metrics and Gates

```text
{config["gates"]}
```

## 5. Results

```text
{mean_table}
```

## 6. Interpretation

Agents with non-zero hardening scores pass all V2.1 hardening gates.

## 7. Failure Cases

Failures should be read directly from hardening_summary.csv. A zero hardening score means at least one hardening gate failed.

## 8. Claim Boundary

Supported:
- delayed relation-chain diagnostic hardened against fixed-template shortcuts.

Unsupported:
- real geotechnical time-series modeling.
- unrestricted temporal relation discovery.
- deployment-ready engineering safety AI.
"""
    audit = """# V2.1 Self-Audit

## What this hardening improves

- Tests variable delay.
- Tests false temporal shortcut.
- Tests multi-link temporal edit.
- Requires temporal audit with time indexes.
- Reduces fixed-template false positives.

## Remaining weaknesses

- Candidate links are still predefined.
- Delay values are still from a small finite set.
- Toy world remains deterministic or near-deterministic.
- No real monitoring uncertainty model.
- No numerical slope mechanics.
- No real deployment claim.

## False positive risks

- Agent may learn candidate-link scoring heuristics.
- Agent may still exploit generator regularities.
- Audit strings may be formatted correctly without deep relation discovery.

## Next after V2.1

Do not jump to real deployment.
Potential V3: partial observability + missing sensors + delayed noisy observations + takeover threshold.
"""
    Path("reports/V2_1_HARDENING_REPORT.md").write_text(report, encoding="utf-8")
    Path("reports/V2_1_SELF_AUDIT.md").write_text(audit, encoding="utf-8")
    print(df.groupby("agent")["hardening_gated_score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
