from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def audit_summary(summary_path: str = "results/summary.csv") -> str:
    df = pd.read_csv(summary_path)
    grouped = df.groupby("agent").mean(numeric_only=True)

    def score_for(agent: str) -> float:
        if agent not in grouped.index:
            return 0.0
        return float(grouped.loc[agent, "gated_slope_score"])

    findings: list[str] = []
    findings.append("# Self Audit: Slope Relation Toy")
    findings.append("")
    findings.append("Date: 2026-04-28")
    findings.append("")
    findings.append("## Verdict")
    findings.append("")
    findings.append(
        "The project is useful as a relation-chain diagnostic, but it is not evidence of a deployable engineering AI system."
    )
    findings.append("")
    findings.append("## Checks")
    findings.append("")

    relation_score = score_for("relation_chain")
    surface_score = score_for("surface")
    generic_score = score_for("generic_review")
    structural_score = score_for("structural_memory")
    learned_score = score_for("learned_links")

    findings.append(f"- `relation_chain` gated score: {relation_score:.3f}")
    findings.append(f"- `surface` gated score: {surface_score:.3f}")
    findings.append(f"- `structural_memory` gated score: {structural_score:.3f}")
    findings.append(f"- `learned_links` gated score: {learned_score:.3f}")
    findings.append(f"- `generic_review` gated score: {generic_score:.3f}")
    findings.append("")

    findings.append("## Strengths")
    findings.append("")
    findings.append("- Surface-cue fitting is rejected by OOD and spurious-attack tests.")
    findings.append("- Generic engineering wording is rejected by review scoring and action-point consistency.")
    findings.append("- Structural memory is separated from editable/auditable relation-chain internalization.")
    findings.append("- `learned_links` tests whether an induced editable relation table can pass the same gates.")
    findings.append("- Irrelevant-link rejection checks that hand-picked distractor links remain disabled.")
    findings.append("- Noisy-observation testing checks basic robustness to monitoring-field corruption.")
    findings.append("- The relation-chain model supports counterfactual checks, relation-link edits, and relation audit.")
    findings.append("")

    findings.append("## Known Weaknesses")
    findings.append("")
    findings.append(
        "- `relation_chain` is an oracle-style model: it shares the hand-written relation logic with the environment."
    )
    findings.append(
        "- `learned_links` learns only two predefined candidate links; it does not discover an unrestricted relation graph."
    )
    findings.append("- The toy world is deterministic and far simpler than real slope engineering.")
    findings.append("- Thresholds are manually chosen and should not be treated as calibrated safety thresholds.")
    findings.append("- Monitoring noise is single-step field corruption; there is still no time dynamics or numerical geotechnical model.")
    findings.append("- The noisy-observation gate is set to 0.7 because a 10% monitoring flip changes many optimal actions in this toy.")
    findings.append("- `structural_memory` is stronger than surface cues, but it is still a table-memory baseline.")
    findings.append("")

    findings.append("## False Positive Risks")
    findings.append("")
    findings.append("- A model could memorize the fixed review-consistency cases.")
    findings.append("- A model could pass by learning the small candidate-link set without learning broader engineering structure.")
    findings.append("- A template could pass review fields if it encodes the same fixed action-point map.")
    findings.append("- Passing this toy does not imply readiness for real plan approval or risk control.")
    findings.append("")

    findings.append("## Next Required Improvement")
    findings.append("")
    findings.append("Noisy monitoring is now single-step only.")
    findings.append("Next: add time-dependent monitoring so learned links must handle delayed effects.")
    findings.append("")
    findings.append("## Claim Boundary")
    findings.append("")
    findings.append("Supported:")
    findings.append("")
    findings.append("- The current harness distinguishes relation-chain reasoning from surface labels and generic review text.")
    findings.append("- A minimal predefined-link learner can pass the same gates after observing enough samples.")
    findings.append("")
    findings.append("Not supported:")
    findings.append("")
    findings.append("- Autonomous discovery of slope engineering relations.")
    findings.append("- Unrestricted relation-graph discovery.")
    findings.append("- Real-world geotechnical correctness.")
    findings.append("- Any safety-critical deployment claim.")
    findings.append("")
    return "\n".join(findings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--output", default="reports/self_audit.md")
    args = parser.parse_args()

    report = audit_summary(args.summary)
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
