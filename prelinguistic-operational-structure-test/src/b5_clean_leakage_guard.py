from __future__ import annotations

from typing import Any

from .b5_clean_episode_view import FORBIDDEN_POLICY_KEYS, find_forbidden_key_paths


def guard_policy_input(model_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    forbidden = set(config.get("b5_clean", {}).get("forbidden_policy_keys", FORBIDDEN_POLICY_KEYS))
    paths = find_forbidden_key_paths(model_input, forbidden)
    report = {
        "model_input_leakage_count": len(paths),
        "forbidden_key_paths": ";".join(paths),
        "fail_fast_triggered": bool(paths and config.get("b5_clean", {}).get("fail_fast_on_leakage", False)),
    }
    if report["fail_fast_triggered"]:
        raise AssertionError(f"Forbidden keys in model_input: {paths}")
    return report


def guard_policy_output(policy_output: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    provenance = policy_output.get("provenance", {})
    oracle_used = bool(
        provenance.get("oracle_plan_used", False)
        or provenance.get("oracle_closed_loop_plan_used", False)
        or provenance.get("oracle_trace_update_used", False)
        or provenance.get("oracle_feedback_revision_used", False)
        or provenance.get("oracle_value_used", False)
    )
    return {
        "policy_output_oracle_usage_rate": 1.0 if oracle_used else 0.0,
        "forbidden_key_paths": "",
        "fail_fast_triggered": False,
    }


def guard_baseline_access(baseline_name: str, baseline_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    has_oracle_view = "oracle_baseline_view" in baseline_input
    violation = has_oracle_view and baseline_name != "oracle_closed_loop_baseline"
    report = {
        "oracle_baseline_access_violation_count": 1 if violation else 0,
        "forbidden_key_paths": "oracle_baseline_view" if violation else "",
        "fail_fast_triggered": bool(violation and config.get("b5_clean", {}).get("fail_fast_on_leakage", False)),
    }
    if report["fail_fast_triggered"]:
        raise AssertionError(f"Non-oracle baseline accessed oracle view: {baseline_name}")
    return report


def build_leakage_record(episode_id: int, stage: str, leakage_paths: list[str], source: str, model: str = "") -> dict[str, Any]:
    return {
        "episode_id": int(episode_id),
        "model": model,
        "stage": stage,
        "source": source,
        "leakage_count": len(leakage_paths),
        "forbidden_key_paths": ";".join(leakage_paths),
        "fail_fast_triggered": 0,
    }
