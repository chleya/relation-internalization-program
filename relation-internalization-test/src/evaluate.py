from copy import deepcopy

from .env import RelationWorld
from .features import all_contexts
from .metrics import (
    adaptation_steps,
    compute_average_reward,
    compute_success_rate,
    gated_internalization_score,
    edit_locality,
    edit_success,
    internalization_score,
)


def run_phase(agent, env, n_steps: int, learn: bool = True) -> list[dict]:
    records = []
    for _ in range(n_steps):
        context = env.sample_context()
        action = agent.act(context)
        record = env.step(action, context=context)
        if learn:
            agent.observe(record["context"], record["action"], record["resource"], record["reward"])
        records.append(record)
    return records


def evaluate_base(agent, seed: int, n_steps: int) -> dict:
    env = RelationWorld("base", seed=seed)
    records = run_phase(agent, env, n_steps, learn=True)
    return {"base_reward": compute_average_reward(records), "base_success": compute_success_rate(records), "records": records}


def evaluate_reversal(agent, seed: int, pre_steps: int, post_steps: int, threshold: float, window: int) -> dict:
    env = RelationWorld("base", seed=seed + 101)
    pre_records = run_phase(agent, env, pre_steps, learn=True)
    env.switch_regime("reversal")
    post_records = run_phase(agent, env, post_steps, learn=True)
    rewards = [r["reward"] for r in post_records]
    return {
        "reversal_pre_reward": compute_average_reward(pre_records),
        "reversal_post_reward": compute_average_reward(post_records),
        "reversal_post_success": compute_success_rate(post_records),
        "reversal_adaptation_steps": adaptation_steps(rewards, threshold=threshold, window=window),
        "reversal_post_steps": post_steps,
        "records": pre_records + post_records,
    }


def evaluate_frozen_reversal(agent, seed: int, train_steps: int, test_steps: int) -> dict:
    train_env = RelationWorld("base", seed=seed + 151)
    train_records = run_phase(agent, train_env, train_steps, learn=True)
    test_env = RelationWorld("reversal", seed=seed + 152)
    test_records = run_phase(agent, test_env, test_steps, learn=False)
    return {
        "frozen_reversal_success": compute_success_rate(test_records),
        "frozen_reversal_reward": compute_average_reward(test_records),
        "records": train_records + test_records,
    }


def evaluate_ood(agent, seed: int, train_steps: int, test_steps: int) -> dict:
    train_env = RelationWorld("base", seed=seed + 202)
    train_records = run_phase(agent, train_env, train_steps, learn=True)
    test_env = RelationWorld("ood", seed=seed + 303, ood=True)
    test_records = run_phase(agent, test_env, test_steps, learn=False)
    return {
        "ood_reward": compute_average_reward(test_records),
        "ood_success": compute_success_rate(test_records),
        "records": train_records + test_records,
    }


def evaluate_spurious_attack(agent, seed: int, train_steps: int) -> dict:
    train_env = RelationWorld("base", seed=seed + 333)
    train_records = run_phase(agent, train_env, train_steps, learn=True)
    test_env = RelationWorld("base", seed=seed + 334)
    attack_contexts = adversarial_spurious_contexts(test_env)
    test_records = []
    resource_correct = 0
    for context in attack_contexts:
        action = agent.act(context)
        record = test_env.step(action, context=context)
        test_records.append(record)
        resource_correct += int(infer_agent_resource(agent, context) == record["resource"])
    return {
        "spurious_robustness": compute_success_rate(test_records),
        "spurious_resource_accuracy": resource_correct / len(test_records),
        "records": train_records + test_records,
    }


def adversarial_spurious_contexts(env: RelationWorld) -> list[dict]:
    contexts = []
    for texture in ["A", "B", "C"]:
        for wet in ["dry", "wet"]:
            base = {"texture": texture, "wet": wet, "color": "color_00", "odor": "odor_00"}
            resource = env.get_resource(base)
            if resource == "food":
                color, odor = "color_01", "odor_01"
            elif resource == "poison":
                color, odor = "color_00", "odor_00"
            else:
                color, odor = "color_00", "odor_01"
            for alt_color in [color, "color_04"]:
                for alt_odor in [odor, "odor_04"]:
                    contexts.append({"texture": texture, "wet": wet, "color": alt_color, "odor": alt_odor})
    return contexts


def evaluate_counterfactual(agent, seed: int) -> dict:
    env = RelationWorld("base", seed=seed + 404)
    contexts = all_contexts()
    correct = 0
    total = 0
    for context in contexts:
        altered = dict(context)
        altered["wet"] = "wet" if context["wet"] == "dry" else "dry"
        expected_resource = env.get_resource(altered)
        actual_resource = infer_agent_resource(agent, altered)
        correct += int(actual_resource == expected_resource)
        total += 1
    return {"counterfactual_accuracy": correct / total if total else 0.0}


def infer_agent_resource(agent, context: dict) -> str:
    if hasattr(agent, "infer_resource"):
        return agent.infer_resource(context)
    return "food" if agent.act(context) == "eat" else "poison"


def evaluate_edit(agent, seed: int) -> dict:
    target_contexts = [
        {"texture": "A", "wet": "dry", "color": color, "odor": odor}
        for color in ["color_00", "color_04", "color_08", "color_12"]
        for odor in ["odor_00", "odor_04", "odor_08", "odor_12"]
    ]
    non_target_contexts = [ctx for ctx in all_contexts() if not (ctx["texture"] == "A" and ctx["wet"] == "dry")]
    before_target = [agent.act(ctx) for ctx in target_contexts]
    before_non_target = [agent.act(ctx) for ctx in non_target_contexts]
    before_target_resources = [infer_agent_resource(agent, ctx) for ctx in target_contexts]
    before_non_target_resources = [infer_agent_resource(agent, ctx) for ctx in non_target_contexts]
    supported = agent.edit_rule({"texture": "A", "wet": "dry"}, "poison")
    after_target = [agent.act(ctx) for ctx in target_contexts]
    after_non_target = [agent.act(ctx) for ctx in non_target_contexts]
    after_target_resources = [infer_agent_resource(agent, ctx) for ctx in target_contexts]
    after_non_target_resources = [infer_agent_resource(agent, ctx) for ctx in non_target_contexts]
    return {
        "edit_supported": supported,
        "edit_success": edit_success(before_target, after_target, target_contexts) if supported else 0.0,
        "edit_locality": edit_locality(before_non_target, after_non_target, non_target_contexts),
        "edit_resource_success": resource_edit_success(before_target_resources, after_target_resources, "poison") if supported else 0.0,
        "edit_resource_locality": resource_edit_locality(before_non_target_resources, after_non_target_resources),
    }


def resource_edit_success(before_resources: list[str], after_resources: list[str], target_resource: str) -> float:
    if not after_resources:
        return 0.0
    changed_correctly = sum(
        1 for before, after in zip(before_resources, after_resources) if before != target_resource and after == target_resource
    )
    return changed_correctly / len(after_resources)


def resource_edit_locality(before_resources: list[str], after_resources: list[str]) -> float:
    if not after_resources:
        return 1.0
    unchanged = sum(1 for before, after in zip(before_resources, after_resources) if before == after)
    return unchanged / len(after_resources)


def evaluate_edit_reversal(agent, seed: int) -> dict:
    contexts = all_contexts()
    before = [agent.act(ctx) for ctx in contexts]
    edits = [
        ({"texture": "A", "wet": "dry"}, "poison"),
        ({"texture": "A", "wet": "wet"}, "food"),
        ({"texture": "B"}, "food"),
    ]
    supported = all(agent.edit_rule(condition, outcome) for condition, outcome in edits)
    after = [agent.act(ctx) for ctx in contexts]
    reversal_env = RelationWorld("reversal", seed=seed + 505)
    correct = 0
    for ctx, action in zip(contexts, after):
        expected = "eat" if reversal_env.get_resource(ctx) == "food" else "avoid"
        correct += int(action == expected)
    changed = sum(1 for old, new in zip(before, after) if old != new) / len(contexts)
    return {
        "edit_reversal_supported": supported,
        "edit_reversal_success": correct / len(contexts) if supported else 0.0,
        "edit_reversal_changed_fraction": changed if supported else 0.0,
    }


def evaluate_relation_shuffle_audit(agent, seed: int) -> dict:
    contexts = all_contexts()
    env = RelationWorld("base", seed=seed + 606)
    before_acc = resource_accuracy(agent, env, contexts)
    supported = hasattr(agent, "shuffle_rule_outcomes") and agent.shuffle_rule_outcomes(seed=seed + 607)
    after_acc = resource_accuracy(agent, env, contexts) if supported else before_acc
    return {
        "relation_shuffle_supported": supported,
        "relation_resource_accuracy": before_acc,
        "relation_shuffled_resource_accuracy": after_acc,
        "relation_shuffle_drop": max(0.0, before_acc - after_acc) if supported else 0.0,
    }


def evaluate_relation_table_alignment(agent) -> dict:
    expected = {
        (("texture", "A"), ("wet", "dry")): "food",
        (("texture", "A"), ("wet", "wet")): "poison",
        (("texture", "B"), ("wet", "dry")): "poison",
        (("texture", "B"), ("wet", "wet")): "poison",
        (("texture", "C"), ("wet", "dry")): "neutral",
        (("texture", "C"), ("wet", "wet")): "neutral",
    }
    rows = agent.describe_relations()
    exact = {}
    for row in rows:
        condition = row.get("condition", {})
        if set(condition) == {"texture", "wet"}:
            exact[tuple(sorted(condition.items()))] = row.get("outcome")
    covered = [key for key in expected if key in exact]
    correct = [key for key in covered if exact[key] == expected[key]]
    coverage = len(covered) / len(expected)
    accuracy = len(correct) / len(covered) if covered else 0.0
    return {
        "relation_table_coverage": coverage,
        "relation_table_accuracy": accuracy,
        "relation_table_alignment": coverage * accuracy,
    }


def resource_accuracy(agent, env: RelationWorld, contexts: list[dict]) -> float:
    correct = 0
    for context in contexts:
        correct += int(infer_agent_resource(agent, context) == env.get_resource(context))
    return correct / len(contexts) if contexts else 0.0


def evaluate_all(agent, config: dict) -> dict:
    env_cfg = config["env"]
    eval_cfg = config["evaluation"]
    seed = config.get("seed", 0)

    base_agent = deepcopy(agent)
    base = evaluate_base(base_agent, seed, env_cfg["n_steps_base"])

    reversal_agent = deepcopy(agent)
    reversal = evaluate_reversal(
        reversal_agent,
        seed,
        env_cfg["n_steps_reversal_pre"],
        env_cfg["n_steps_reversal_post"],
        eval_cfg["adaptation_threshold"],
        eval_cfg["adaptation_window"],
    )

    frozen_reversal_agent = deepcopy(agent)
    frozen_reversal = evaluate_frozen_reversal(
        frozen_reversal_agent,
        seed,
        env_cfg["n_steps_reversal_pre"],
        env_cfg["n_steps_ood_test"],
    )

    ood_agent = deepcopy(agent)
    ood = evaluate_ood(ood_agent, seed, env_cfg["n_steps_ood_train"], env_cfg["n_steps_ood_test"])
    spurious_agent = deepcopy(agent)
    spurious = evaluate_spurious_attack(spurious_agent, seed, env_cfg["n_steps_ood_train"])
    counterfactual = evaluate_counterfactual(ood_agent, seed)
    shuffle_audit = evaluate_relation_shuffle_audit(deepcopy(ood_agent), seed)
    table_alignment = evaluate_relation_table_alignment(ood_agent)
    edit = evaluate_edit(ood_agent, seed)
    edit_reversal = evaluate_edit_reversal(ood_agent, seed)

    metrics = {
        "agent": agent.name,
        "seed": seed,
        **{k: v for k, v in base.items() if k != "records"},
        **{k: v for k, v in reversal.items() if k != "records"},
        **{k: v for k, v in frozen_reversal.items() if k != "records"},
        **{k: v for k, v in ood.items() if k != "records"},
        **{k: v for k, v in spurious.items() if k != "records"},
        **counterfactual,
        **edit,
        **edit_reversal,
        **shuffle_audit,
        **table_alignment,
    }
    metrics["internalization_score"] = internalization_score(metrics)
    metrics["gated_internalization_score"] = gated_internalization_score(metrics)
    records = []
    for phase, phase_records in [
        ("base", base["records"]),
        ("reversal", reversal["records"]),
        ("frozen_reversal", frozen_reversal["records"]),
        ("ood", ood["records"]),
        ("spurious_attack", spurious["records"]),
    ]:
        for idx, record in enumerate(phase_records):
            row = dict(record)
            row["phase"] = phase
            row["step"] = idx
            row.update(row.pop("context"))
            records.append(row)
    return {"metrics": metrics, "records": records, "relations": ood_agent.describe_relations()}
