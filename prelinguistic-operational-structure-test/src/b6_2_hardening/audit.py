from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any

from .policy import b62_policy


FORBIDDEN = ["evaluator_ground_truth", "oracle_baseline_view", "expected_decision", "expected_action", "condition", "episode_type"]


def policy_source_audit() -> dict[str, Any]:
    source = inspect.getsource(b62_policy)
    hits = [token for token in FORBIDDEN if token in source]
    return {"forbidden_reference_count": len(hits), "forbidden_references": hits, "policy_uses_model_input_only": len(hits) == 0}


def poisoned_invariance_check(episode: dict[str, Any]) -> bool:
    base = b62_policy(episode, {})
    poisoned = copy.deepcopy(episode)
    poisoned["metadata"]["condition"] = "poisoned"
    poisoned["evaluator_ground_truth"]["condition"] = "poisoned"
    poisoned["evaluator_ground_truth"]["expected_action"] = None
    poisoned["oracle_baseline_view"]["expected_action"] = None
    changed = b62_policy(poisoned, {})
    return base == changed


def audit_for_episode(episode: dict[str, Any]) -> dict[str, Any]:
    source = policy_source_audit()
    return {**source, "poisoned_ground_truth_invariance_pass": int(poisoned_invariance_check(episode))}


def write_audit(path: Path, summary: list[dict[str, Any]]) -> None:
    audit = {
        "summary_rows": len(summary),
        "constant_one_metric_regression": "B6.2 includes wrong-decision tests and no empty-set metric defaults to full credit.",
        "n_ood_note": "B6 clean had n_ood config without an active OOD split. B6.2 uses explicit condition-level stress splits.",
        "audit_derived_limitations": [
            "B6/B6.1 clean masks exposed strong operational cues.",
            "previous_trace_state.region remains a strong target prior and is now stressed through wrong/missing/ambiguous trace conditions.",
            "Hidden indirect causal path discovery is not proven.",
            "Delayed credit assignment remains toy delayed-outcome history, not real-world causal credit.",
        ],
    }
    path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

