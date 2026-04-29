from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics_uncertainty import evaluate_v3_agent
from .utils import load_yaml


AGENTS = [
    "surface_temporal",
    "structural_memory_temporal",
    "instant_relation_chain",
    "delayed_relation_chain",
    "learned_delayed_links",
    "uncertainty_aware_delayed_links",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_uncertainty.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)

    rows = []
    for agent_name in AGENTS:
        for seed in config["seeds"]:
            rows.append({"agent": agent_name, "seed": seed, **evaluate_v3_agent(agent_name, seed, config)})

    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/v3_uncertainty_summary.csv", index=False)
    mean_table = df.groupby("agent").mean(numeric_only=True).round(3).to_string()

    report = f"""# V3 Uncertainty and Takeover Report

## 1. Motivation

V2.1 showed that temporal prediction is not temporal relation internalization. V3 tests the next boundary: relation use under missing sensors, delayed noisy observations, and takeover conditions.

## 2. Setup

The toy temporal chain remains the V2/V2.1 delayed slope relation chain. V3 corrupts observations with missing values, noise, and delayed reports, then evaluates whether an agent should act automatically or trigger takeover.

## 3. Metrics

- noisy_action_success
- takeover_precision
- takeover_recall
- unsafe_automation_rate
- uncertain_relation_audit_score
- gated_v3_score

## 4. Gates

```text
{config["gates"]}
```

## 5. Results

```text
{mean_table}
```

## 6. Interpretation

Non-zero gated_v3_score requires accurate action, high takeover precision/recall, low unsafe automation, and concrete uncertain relation audit.

## 7. Claim Boundary

Supported:
- toy diagnostic for uncertainty-aware temporal relation-chain takeover.

Unsupported:
- real geotechnical time-series modeling.
- deployment-ready engineering safety AI.
- unrestricted temporal relation discovery.
"""
    audit = """# V3 Self-Audit

## What V3 improves

- Adds missing sensor observations.
- Adds delayed noisy observations.
- Adds takeover as an explicit action.
- Measures unsafe automation.
- Requires uncertain relation audit.

## Remaining weaknesses

- Still a toy diagnostic.
- Candidate physical links are predefined.
- Noise model is simple and synthetic.
- Takeover ground truth is rule-defined.
- No real geotechnical monitoring data.
- No deployment claim.

## False positive risks

- An agent may learn the uncertainty generator rather than robust engineering uncertainty.
- Audit text may still be formatted correctly without real-world domain understanding.
- Oracle delayed chain has hand-coded relation structure.

## Next after V3

Only after V3 is stable should the project consider a small V4 engineering review case, with no safety deployment claim.
"""
    Path("reports/V3_UNCERTAINTY_REPORT.md").write_text(report, encoding="utf-8")
    Path("reports/V3_SELF_AUDIT.md").write_text(audit, encoding="utf-8")
    print(df.groupby("agent")["gated_v3_score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()

