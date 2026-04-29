from __future__ import annotations

import random
from statistics import mean
from typing import Any

from .partial_agents import AgentDecision, inspection_values, make_partial_agent, relation_audit
from .partial_env import CRITICAL_FIELDS, NONCRITICAL_FIELDS, PartialProcessWorld, hard_case_state, mask_fields


def is_inspection(decision: AgentDecision) -> bool:
    return decision.action == "inspect"


def needs_relation_inspection(state: dict[str, str]) -> bool:
    if any(item.get("reason") == "conflicting_relation" for item in relation_audit(state)):
        return True
    if state.get("risk") in {"high", "low"}:
        return False
    if state.get("crack") in {"open", "closed"}:
        return False
    if state.get("displacement") in {"high", "normal"}:
        return False
    return any(state.get(field) == "unknown" for field in CRITICAL_FIELDS)


def critical_missing_cases() -> list[dict[str, str]]:
    base = hard_case_state()
    cases = []
    for field in ["pore_pressure", "anchoring", "displacement", "crack", "risk"]:
        case = dict(base)
        for uncertain in ["pore_pressure", "anchoring", "displacement", "crack", "risk"]:
            case[uncertain] = "unknown"
        case[field] = "unknown"
        cases.append(case)
    return cases


def noncritical_missing_cases() -> list[dict[str, str]]:
    base = hard_case_state()
    safe = PartialProcessWorld().evaluate(
        {
            "rainfall": "low",
            "drainage": "good",
            "anchoring": "present",
            "contractor_report": "clear",
            "weather_label": "clear",
            "surface_warning": "none",
        }
    )
    cases = []
    for field in NONCRITICAL_FIELDS:
        cases.append(mask_fields(base, [field]))
        cases.append(mask_fields(safe, [field]))
    return cases


def mixed_observability_cases() -> list[dict[str, str]]:
    high = hard_case_state()
    low = PartialProcessWorld().evaluate(
        {
            "rainfall": "low",
            "drainage": "good",
            "anchoring": "present",
            "contractor_report": "clear",
            "weather_label": "clear",
            "surface_warning": "none",
        }
    )
    return [
        mask_fields(high, ["pore_pressure", "displacement"]),
        mask_fields(high, ["pore_pressure", "anchoring", "displacement"]),
        mask_fields(low, ["pore_pressure", "anchoring"]),
        mask_fields(low, ["pore_pressure", "anchoring", "contractor_report"]),
    ]


def conflict_cases() -> list[tuple[dict[str, str], str]]:
    base = hard_case_state()
    pore_conflict = dict(base)
    pore_conflict["pore_pressure"] = "normal"
    pore_conflict["anchoring"] = "absent"
    pore_conflict["displacement"] = "high"

    anchoring_conflict = dict(base)
    anchoring_conflict["anchoring"] = "present"
    anchoring_conflict["displacement"] = "high"

    crack_conflict = dict(base)
    crack_conflict["displacement"] = "normal"
    crack_conflict["crack"] = "open"
    return [
        (pore_conflict, "PorePressure -> Displacement"),
        (anchoring_conflict, "Anchoring -> DisplacementDown"),
        (crack_conflict, "Displacement -> Crack"),
    ]


def critical_missing_inspection_recall(agent_name: str) -> float:
    agent = make_partial_agent(agent_name)
    decisions = [agent.decide(case) for case in critical_missing_cases()]
    return mean(1.0 if is_inspection(decision) else 0.0 for decision in decisions)


def noncritical_missing_no_inspect_rate(agent_name: str) -> float:
    agent = make_partial_agent(agent_name)
    decisions = [agent.decide(case) for case in noncritical_missing_cases()]
    return mean(1.0 if not is_inspection(decision) else 0.0 for decision in decisions)


def inspection_precision_and_unnecessary(agent_name: str) -> tuple[float, float]:
    agent = make_partial_agent(agent_name)
    necessary = [(case, True) for case in critical_missing_cases()]
    unnecessary = [(case, False) for case in noncritical_missing_cases() + mixed_observability_cases()]
    inspected_needed = 0
    inspected_total = 0
    unnecessary_inspects = 0
    for case, is_needed in necessary + unnecessary:
        decision = agent.decide(case)
        if is_inspection(decision):
            inspected_total += 1
            if is_needed:
                inspected_needed += 1
            else:
                unnecessary_inspects += 1
    precision = inspected_needed / inspected_total if inspected_total else 0.0
    unnecessary_rate = unnecessary_inspects / len(unnecessary)
    return precision, unnecessary_rate


def unsafe_automation_rate(agent_name: str) -> float:
    agent = make_partial_agent(agent_name)
    cases = critical_missing_cases() + [case for case, _ in conflict_cases()]
    unsafe = []
    for case in cases:
        decision = agent.decide(case)
        unsafe.append(1.0 if needs_relation_inspection(case) and not is_inspection(decision) else 0.0)
    return mean(unsafe)


def conflict_localization_accuracy(agent_name: str) -> float:
    agent = make_partial_agent(agent_name)
    hits = []
    for case, expected_link in conflict_cases():
        decision = agent.decide(case)
        links = {str(item.get("link")) for item in decision.audit}
        text = " ".join(str(item) for item in decision.audit).lower()
        hits.append(1.0 if expected_link in links and "state incomplete" not in text else 0.0)
    return mean(hits)


def non_oracle_inspection_update_accuracy(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_partial_agent(agent_name)
    env = PartialProcessWorld(seed=seed + 10_000, inspection_noise=float(config["inspection_noise"]))
    rng = random.Random(seed + 11_000)
    hits = []
    for _ in range(int(config["non_oracle_trials"])):
        true_state = env.sample_state()
        field = rng.choice(CRITICAL_FIELDS)
        observed = mask_fields(true_state, [field])
        if field in {"pore_pressure", "anchoring"}:
            observed["displacement"] = "unknown"
            observed["crack"] = "unknown"
            observed["risk"] = "unknown"
        elif field == "displacement":
            observed["crack"] = "unknown"
            observed["risk"] = "unknown"
        elif field == "crack":
            observed["risk"] = "unknown"
        decision = agent.decide(observed)
        inspect_field = decision.inspect_field if is_inspection(decision) and decision.inspect_field else field
        updated, result = env.inspect(true_state, observed, inspect_field)
        changed_fields = [key for key in updated if updated.get(key) != observed.get(key)]
        only_one_field_changed = changed_fields == [inspect_field]
        hits.append(1.0 if only_one_field_changed and result.accurate else 0.0)
    return mean(hits)


def cost_adjusted_success(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_partial_agent(agent_name)
    env = PartialProcessWorld(seed=seed + 20_000, inspection_noise=float(config["inspection_noise"]))
    rng = random.Random(seed + 21_000)
    budget = int(config["inspection_budget"])
    scores = []
    scenarios = ["critical", "noncritical", "mixed"]
    for _ in range(int(config["cost_trials"])):
        true_state = env.sample_state()
        scenario = rng.choice(scenarios)
        if scenario == "critical":
            field = rng.choice(["pore_pressure", "anchoring", "displacement", "crack", "risk"])
            observed = mask_fields(true_state, [field])
            if field != "risk":
                observed["risk"] = "unknown"
            if field not in {"crack", "risk"}:
                observed["crack"] = "unknown"
        elif scenario == "noncritical":
            observed = mask_fields(true_state, [rng.choice(NONCRITICAL_FIELDS)])
        else:
            observed = mask_fields(true_state, ["pore_pressure", "displacement"])
            observed["risk"] = true_state["risk"]

        score = 0.0
        decision = agent.decide(observed)
        if is_inspection(decision):
            score -= float(config["inspection_cost"])
            if budget > 0 and decision.inspect_field:
                budget -= 1
                observed, _ = env.inspect(true_state, observed, decision.inspect_field)
                decision = agent.decide(observed)
            else:
                score -= float(config["budget_exhaustion_penalty"])

        if needs_relation_inspection(observed) and not is_inspection(decision):
            score -= float(config["unsafe_automation_penalty"])

        score += 1.0 if decision.action == env.optimal_action(true_state) else 0.0
        scores.append(score)
    return mean(scores)


def evaluate_r2_1_agent(agent_name: str, seed: int, config: dict[str, Any]) -> dict[str, float]:
    precision, unnecessary = inspection_precision_and_unnecessary(agent_name)
    return {
        "critical_missing_inspection_recall": critical_missing_inspection_recall(agent_name),
        "noncritical_missing_no_inspect_rate": noncritical_missing_no_inspect_rate(agent_name),
        "inspection_precision": precision,
        "unnecessary_inspection_rate": unnecessary,
        "unsafe_automation_rate": unsafe_automation_rate(agent_name),
        "conflict_localization_accuracy": conflict_localization_accuracy(agent_name),
        "cost_adjusted_success": cost_adjusted_success(agent_name, seed, config),
        "non_oracle_inspection_update_accuracy": non_oracle_inspection_update_accuracy(agent_name, seed, config),
    }


def r2_1_gated_score(metrics: dict[str, float], gates: dict[str, float], baseline_cost_adjusted_success: float) -> float:
    required = [
        metrics["critical_missing_inspection_recall"] >= gates["critical_missing_inspection_recall"],
        metrics["noncritical_missing_no_inspect_rate"] >= gates["noncritical_missing_no_inspect_rate"],
        metrics["inspection_precision"] >= gates["inspection_precision"],
        metrics["unsafe_automation_rate"] <= gates["unsafe_automation_rate"],
        metrics["unnecessary_inspection_rate"] <= gates["unnecessary_inspection_rate"],
        metrics["conflict_localization_accuracy"] >= gates["conflict_localization_accuracy"],
        metrics["non_oracle_inspection_update_accuracy"] >= gates["non_oracle_inspection_update_accuracy"],
        metrics["cost_adjusted_success"] >= baseline_cost_adjusted_success + gates["cost_adjusted_success_margin"],
    ]
    if not all(required):
        return 0.0
    safety_score = 1.0 - metrics["unsafe_automation_rate"]
    efficiency_score = 1.0 - metrics["unnecessary_inspection_rate"]
    return mean(
        [
            metrics["critical_missing_inspection_recall"],
            metrics["noncritical_missing_no_inspect_rate"],
            metrics["inspection_precision"],
            safety_score,
            efficiency_score,
            metrics["conflict_localization_accuracy"],
            metrics["non_oracle_inspection_update_accuracy"],
            min(1.0, metrics["cost_adjusted_success"]),
        ]
    )


def r3_multi_field_cases() -> list[tuple[dict[str, str], dict[str, str], str]]:
    env = PartialProcessWorld(seed=31)
    high = hard_case_state()
    safe = env.evaluate(
        {
            "rainfall": "low",
            "drainage": "good",
            "anchoring": "present",
            "contractor_report": "clear",
            "weather_label": "clear",
            "surface_warning": "none",
        }
    )
    cases = [
        (high, mask_fields(high, ["contractor_report", "weather_label", "pore_pressure", "displacement", "crack", "risk"]), "risk"),
        (high, mask_fields(high, ["surface_warning", "anchoring", "displacement", "crack", "risk"]), "risk"),
        (safe, mask_fields(safe, ["contractor_report", "weather_label", "pore_pressure", "displacement", "crack", "risk"]), "risk"),
        (safe, mask_fields(safe, ["surface_warning", "displacement", "crack", "risk"]), "risk"),
        (high, mask_fields(high, ["contractor_report", "surface_warning", "anchoring", "displacement", "crack", "risk"]), "risk"),
    ]
    return cases


def inspection_target_accuracy(agent_name: str) -> float:
    agent = make_partial_agent(agent_name)
    hits = []
    for _, observed, target in r3_multi_field_cases():
        decision = agent.decide(observed)
        hits.append(1.0 if is_inspection(decision) and decision.inspect_field == target else 0.0)
    return mean(hits)


def information_gain_efficiency(agent_name: str) -> float:
    agent = make_partial_agent(agent_name)
    ratios = []
    for _, observed, _ in r3_multi_field_cases():
        values = inspection_values(observed)
        best = max([value for value in values.values() if value > 0.0], default=0.0)
        decision = agent.decide(observed)
        chosen = values.get(decision.inspect_field or "", 0.0)
        ratios.append(chosen / best if best > 0.0 else 0.0)
    return mean(ratios)


def run_sequential_case(
    agent_name: str,
    true_state: dict[str, str],
    observed: dict[str, str],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    agent = make_partial_agent(agent_name)
    env = PartialProcessWorld(seed=seed, inspection_noise=float(config["inspection_noise"]))
    current = dict(observed)
    max_inspections = int(config["max_inspections_per_case"])
    inspections = 0
    selected_fields: list[str] = []
    accurate_updates = 0
    decision = agent.decide(current)
    unsafe = False
    overinspect = False

    while is_inspection(decision):
        if decision.inspect_field is None:
            overinspect = True
            break
        if inspections >= max_inspections:
            overinspect = True
            break
        if observed_sufficient_after_current(current):
            overinspect = True
        selected_fields.append(decision.inspect_field)
        before = dict(current)
        current, result = env.inspect(true_state, current, decision.inspect_field)
        changed_fields = [key for key in current if current.get(key) != before.get(key)]
        if changed_fields == [decision.inspect_field] and result.accurate:
            accurate_updates += 1
        inspections += 1
        decision = agent.decide(current)

    if needs_relation_inspection(current) and not is_inspection(decision):
        unsafe = True
    final_correct = decision.action == PartialProcessWorld().optimal_action(true_state)
    return {
        "final_correct": final_correct,
        "unsafe": unsafe,
        "overinspect": overinspect,
        "inspections": inspections,
        "selected_fields": selected_fields,
        "accurate_updates": accurate_updates,
        "decision": decision,
    }


def observed_sufficient_after_current(state: dict[str, str]) -> bool:
    return state.get("risk") in {"high", "low"} or state.get("crack") in {"open", "closed"} or state.get("displacement") in {"high", "normal"}


def budgeted_safe_action_rate(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    hits = []
    for idx, (true_state, observed, _) in enumerate(r3_multi_field_cases()):
        result = run_sequential_case(agent_name, true_state, observed, config, seed + idx)
        hits.append(1.0 if result["final_correct"] and not result["unsafe"] and not result["overinspect"] else 0.0)
    return mean(hits)


def r3_unsafe_automation_rate(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    hits = []
    for idx, (true_state, observed, _) in enumerate(r3_multi_field_cases()):
        result = run_sequential_case(agent_name, true_state, observed, config, seed + 100 + idx)
        hits.append(1.0 if result["unsafe"] else 0.0)
    return mean(hits)


def overinspection_rate(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    hits = []
    for idx, (true_state, observed, _) in enumerate(r3_multi_field_cases()):
        result = run_sequential_case(agent_name, true_state, observed, config, seed + 200 + idx)
        hits.append(1.0 if result["overinspect"] or result["inspections"] > int(config["max_inspections_per_case"]) else 0.0)
    return mean(hits)


def sequential_update_accuracy(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    accurate = []
    for idx, (true_state, observed, _) in enumerate(r3_multi_field_cases()):
        result = run_sequential_case(agent_name, true_state, observed, config, seed + 300 + idx)
        if result["inspections"] > 0:
            accurate.append(result["accurate_updates"] / result["inspections"])
    return mean(accurate) if accurate else 0.0


def r3_cost_adjusted_success(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    scores = []
    for idx, (true_state, observed, _) in enumerate(r3_multi_field_cases()):
        result = run_sequential_case(agent_name, true_state, observed, config, seed + 400 + idx)
        score = 1.0 if result["final_correct"] else 0.0
        score -= result["inspections"] * float(config["inspection_cost"])
        if result["unsafe"]:
            score -= float(config["unsafe_automation_penalty"])
        if result["overinspect"]:
            score -= float(config["overinspection_penalty"])
        scores.append(score)
    return mean(scores)


def evaluate_r3_agent(agent_name: str, seed: int, config: dict[str, Any]) -> dict[str, float]:
    return {
        "inspection_target_accuracy": inspection_target_accuracy(agent_name),
        "information_gain_efficiency": information_gain_efficiency(agent_name),
        "budgeted_safe_action_rate": budgeted_safe_action_rate(agent_name, seed, config),
        "unsafe_automation_rate": r3_unsafe_automation_rate(agent_name, seed, config),
        "overinspection_rate": overinspection_rate(agent_name, seed, config),
        "sequential_update_accuracy": sequential_update_accuracy(agent_name, seed, config),
        "cost_adjusted_success": r3_cost_adjusted_success(agent_name, seed, config),
    }


def r3_gated_score(metrics: dict[str, float], gates: dict[str, float], first_missing_cost: float, random_cost: float) -> float:
    required = [
        metrics["inspection_target_accuracy"] >= gates["inspection_target_accuracy"],
        metrics["information_gain_efficiency"] >= gates["information_gain_efficiency"],
        metrics["budgeted_safe_action_rate"] >= gates["budgeted_safe_action_rate"],
        metrics["unsafe_automation_rate"] <= gates["unsafe_automation_rate"],
        metrics["overinspection_rate"] <= gates["overinspection_rate"],
        metrics["sequential_update_accuracy"] >= gates["sequential_update_accuracy"],
        metrics["cost_adjusted_success"] >= first_missing_cost + gates["first_missing_margin"],
        metrics["cost_adjusted_success"] >= random_cost + gates["random_inspect_margin"],
    ]
    if not all(required):
        return 0.0
    return mean(
        [
            metrics["inspection_target_accuracy"],
            metrics["information_gain_efficiency"],
            metrics["budgeted_safe_action_rate"],
            1.0 - metrics["unsafe_automation_rate"],
            1.0 - metrics["overinspection_rate"],
            metrics["sequential_update_accuracy"],
            min(1.0, metrics["cost_adjusted_success"]),
        ]
    )
