from __future__ import annotations

import copy
import inspect
from typing import Any

from .structural_policy import b63_policy
from .trace_repair import detect_trace_conflict, repair_trace_or_request_inspection


FORBIDDEN = [
    "evaluator_ground_truth",
    "oracle_baseline_view",
    "expected_decision",
    "oracle_risk_constrained_plan",
    "true risk label",
    "hidden delayed success",
]


def policy_source_audit() -> dict[str, Any]:
    source = "\n".join(
        [
            inspect.getsource(b63_policy),
            inspect.getsource(detect_trace_conflict),
            inspect.getsource(repair_trace_or_request_inspection),
        ]
    )
    hits = [token for token in FORBIDDEN if token in source]
    return {"forbidden_reference_count": len(hits), "forbidden_references": hits, "policy_uses_model_input_only": len(hits) == 0}


def poisoned_invariance_check(episode: dict[str, Any]) -> bool:
    base = b63_policy(episode, {})
    poisoned = copy.deepcopy(episode)
    poisoned["evaluator_ground_truth"]["expected_action"] = None
    poisoned["evaluator_ground_truth"]["condition"] = "poisoned"
    poisoned["metadata"]["episode_type"] = "poisoned"
    poisoned["oracle_baseline_view"]["expected_action"] = None
    return base == b63_policy(poisoned, {})


def audit_for_episode(episode: dict[str, Any]) -> dict[str, Any]:
    source = policy_source_audit()
    return {**source, "poisoned_ground_truth_invariance_pass": int(poisoned_invariance_check(episode))}
