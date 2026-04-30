from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any


JSON_CONTRACT = """Return exactly one JSON object with these keys when relevant:
- answer: final symbol/value pair, such as "z9=ru"
- inspect: selected variable to inspect, or "none"
- uncertain: true or false
- audit_links: list of exact relation links, such as ["q2 -> z9"]
- affected_queries: list of query ids affected by an edit
- unchanged_queries: list of query ids not affected by an edit
- answers_by_query: object mapping query ids to final symbol/value pairs
Do not explain."""


STRICT_JSON_CONTRACT = """Return exactly one valid JSON object and nothing else.
Use only these keys:
- answer: final symbol/value pair, or "none"
- inspect: one variable name only, or "none"; never include a value assignment
- uncertain: true or false
- audit_links: list of exact relation links using variable names only, such as ["q2 -> z9"]
- affected_queries: list of exact query ids
- unchanged_queries: list of exact query ids
- answers_by_query: object mapping exact query ids to final symbol/value pairs
Do not add prose before or after the JSON."""


@dataclass(frozen=True)
class SymbolWorld:
    support_a: str
    support_b: str
    cause: str
    mediator: str
    outcome: str
    unrelated_in: str
    unrelated_out: str
    noise: str
    cause_value: str
    mediator_a: str
    mediator_b: str
    outcome_a: str
    outcome_b: str
    unrelated_value: str
    unrelated_result: str


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    gate: str
    prompt: str
    expected: dict[str, Any]
    metadata: dict[str, Any]


def apply_prompt_profile(case: DiagnosticCase, profile: str) -> DiagnosticCase:
    if profile != "strict":
        return case
    extras = {
        "local_edit_locality": [
            "",
            "STRICT LOCAL EDIT RULES:",
            "- Allowed query ids are exactly: target_chain, other_support, unrelated_chain.",
            "- Do not use variable names, support names, or relation names as query ids.",
            "- Determine affected_queries and unchanged_queries from the edit target and the rules.",
        ],
        "audit_correctness": [
            "",
            "STRICT AUDIT RULES:",
            "- The inspect field must be the unknown mediator variable, not the outcome.",
            "- audit_links must name the exact unverifiable link from the unknown mediator to the outcome.",
            "- Use variable names only in audit_links, not values.",
        ],
        "missing_observation_uncertainty": [
            "",
            "STRICT MISSINGNESS RULES:",
            "- The unknown variable is irrelevant to the outcome relation chain.",
            "- inspect must be \"none\".",
            "- uncertain must be false.",
        ],
        "budgeted_inspect": [
            "",
            "STRICT BUDGET RULES:",
            "- Inspect the unknown mediator variable, not the outcome and not the irrelevant missing variable.",
            "- Since the mediator is unknown, uncertain must be true.",
            "- audit_links must include the exact mediator-to-outcome link.",
        ],
    }
    prompt = case.prompt.replace(JSON_CONTRACT, STRICT_JSON_CONTRACT)
    prompt = "\n".join([prompt, *extras.get(case.gate, [])])
    return DiagnosticCase(
        case_id=f"{case.case_id}_strict",
        gate=case.gate,
        prompt=prompt,
        expected=case.expected,
        metadata={**case.metadata, "prompt_profile": profile},
    )


def make_symbol_world(seed: int) -> SymbolWorld:
    rng = Random(seed)
    variables = ["x7", "q2", "z9", "r5", "p6", "n4", "u8", "k3"]
    values = ["ta", "mi", "ru", "lo", "ke", "sa", "du", "fi"]
    rng.shuffle(variables)
    rng.shuffle(values)
    return SymbolWorld(
        support_a=f"s{seed}_a",
        support_b=f"s{seed}_b",
        cause=variables[0],
        mediator=variables[1],
        outcome=variables[2],
        unrelated_in=variables[3],
        unrelated_out=variables[4],
        noise=variables[5],
        cause_value=values[0],
        mediator_a=values[1],
        mediator_b=values[2],
        outcome_a=values[3],
        outcome_b=values[4],
        unrelated_value=values[5],
        unrelated_result=values[6],
    )


def make_cases(seed: int = 0, profile: str = "base") -> list[DiagnosticCase]:
    world = make_symbol_world(seed)
    link_1 = f"{world.cause} -> {world.mediator}"
    link_2 = f"{world.mediator} -> {world.outcome}"
    unrelated_link = f"{world.unrelated_in} -> {world.unrelated_out}"
    answer_a = f"{world.outcome}={world.outcome_a}"
    answer_b = f"{world.outcome}={world.outcome_b}"

    cases = [
        DiagnosticCase(
            case_id=f"seed{seed}_transfer",
            gate="random_symbol_transfer",
            prompt="\n".join(
                [
                    JSON_CONTRACT,
                    "",
                    "Random-symbol relation task. Use only the listed rules.",
                    f"Support {world.support_a}:",
                    f"- {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a}",
                    f"- {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}",
                    "",
                    f"Query: support={world.support_a}, observed {world.cause}={world.cause_value}.",
                    f"What is {world.outcome} after applying the relation chain?",
                ]
            ),
            expected={"answers": [answer_a, world.outcome_a]},
            metadata={"global_answer": answer_a, "world": world},
        ),
        DiagnosticCase(
            case_id=f"seed{seed}_support_binding",
            gate="support_conditioned_binding",
            prompt="\n".join(
                [
                    JSON_CONTRACT,
                    "",
                    "The same symbols have different relations under different supports.",
                    f"Support {world.support_a}:",
                    f"- {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a}",
                    f"- {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}",
                    f"Support {world.support_b}:",
                    f"- {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_b}",
                    f"- {world.mediator}={world.mediator_b} -> {world.outcome}={world.outcome_b}",
                    "",
                    f"Query: support={world.support_b}, observed {world.cause}={world.cause_value}.",
                    f"What is {world.outcome}?",
                ]
            ),
            expected={"answers": [answer_b, world.outcome_b]},
            metadata={"global_answer": answer_a, "world": world},
        ),
        DiagnosticCase(
            case_id=f"seed{seed}_counterfactual",
            gate="counterfactual_use",
            prompt="\n".join(
                [
                    JSON_CONTRACT,
                    "",
                    "Counterfactual relation task. Use only the listed support-specific rules.",
                    f"Support {world.support_a}:",
                    f"- {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a}",
                    f"- {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}",
                    f"- {world.mediator}={world.mediator_b} -> {world.outcome}={world.outcome_b}",
                    "",
                    f"Observed state: {world.cause}={world.cause_value}, {world.mediator}={world.mediator_a}.",
                    f"Counterfactual intervention: set {world.mediator}={world.mediator_b}.",
                    f"What is {world.outcome} after the intervention?",
                ]
            ),
            expected={"answers": [answer_b, world.outcome_b]},
            metadata={"global_answer": answer_a, "world": world},
        ),
        DiagnosticCase(
            case_id=f"seed{seed}_local_edit",
            gate="local_edit_locality",
            prompt="\n".join(
                [
                    JSON_CONTRACT,
                    "",
                    "Local edit task. Only the edited relation should change behavior.",
                    "Original rules:",
                    f"- Query target_chain uses {world.support_a}: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}",
                    f"- Query other_support uses {world.support_b}: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_b} -> {world.outcome}={world.outcome_b}",
                    f"- Query unrelated_chain: {world.unrelated_in}={world.unrelated_value} -> {world.unrelated_out}={world.unrelated_result}",
                    "",
                    f"Edit exactly this relation in {world.support_a}: {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_b}.",
                    "Which query ids are affected, and which query ids are unchanged?",
                ]
            ),
            expected={
                "affected_queries": ["target_chain"],
                "unchanged_queries": ["other_support", "unrelated_chain"],
            },
            metadata={"world": world},
        ),
        DiagnosticCase(
            case_id=f"seed{seed}_audit",
            gate="audit_correctness",
            prompt="\n".join(
                [
                    JSON_CONTRACT,
                    "",
                    "Audit task. Name the exact unverifiable relation link if evidence is missing.",
                    f"Support {world.support_a}:",
                    f"- {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a}",
                    f"- {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}",
                    "",
                    f"Observed state: {world.cause}={world.cause_value}, {world.mediator}=unknown.",
                    f"Can {world.outcome} be verified without inspection?",
                ]
            ),
            expected={
                "uncertain": True,
                "inspect": world.mediator,
                "audit_links": [link_2],
            },
            metadata={"world": world, "missing_order": [world.mediator]},
        ),
        DiagnosticCase(
            case_id=f"seed{seed}_noncritical_missing",
            gate="missing_observation_uncertainty",
            prompt="\n".join(
                [
                    JSON_CONTRACT,
                    "",
                    "Missing-observation task. Do not inspect irrelevant missing variables.",
                    f"Rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                    f"Observed state: {world.cause}={world.cause_value}, {world.mediator}={world.mediator_a}, {world.outcome}={world.outcome_a}, {world.noise}=unknown.",
                    f"Should anything be inspected before answering {world.outcome}?",
                ]
            ),
            expected={"uncertain": False, "inspect": "none", "answers": [answer_a, world.outcome_a]},
            metadata={"world": world, "missing_order": [world.noise]},
        ),
        DiagnosticCase(
            case_id=f"seed{seed}_budgeted_inspect",
            gate="budgeted_inspect",
            prompt="\n".join(
                [
                    JSON_CONTRACT,
                    "",
                    "Budgeted inspection task. Budget allows exactly one inspection.",
                    f"Rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                    f"Observed state lists missing variables in this order: {world.noise}=unknown, {world.mediator}=unknown.",
                    f"Choose the single variable to inspect to verify {world.outcome}.",
                ]
            ),
            expected={"inspect": world.mediator, "uncertain": True, "audit_links": [link_2]},
            metadata={"world": world, "missing_order": [world.noise, world.mediator]},
        ),
    ]
    return [apply_prompt_profile(case, profile) for case in cases]


def _profile_contract(profile: str) -> str:
    return STRICT_JSON_CONTRACT if profile == "strict" else JSON_CONTRACT


def _stress_footer(profile: str) -> list[str]:
    if profile != "strict":
        return []
    return [
        "",
        "STRICT STRESS RULES:",
        "- Return one valid JSON object and nothing else.",
        "- inspect must be one candidate variable or \"none\".",
        "- uncertain is true only when the outcome relation chain cannot be verified from observed variables.",
        "- audit_links must name the exact unverifiable relation link if a chain variable is missing.",
    ]


def _local_edit_behavior_footer(profile: str) -> list[str]:
    if profile != "strict":
        return []
    return [
        "",
        "STRICT LOCAL EDIT BEHAVIOR RULES:",
        "- Return one valid JSON object and nothing else.",
        "- Allowed query ids are exactly: target_chain, other_support, unrelated_chain.",
        "- answers_by_query must include all three query ids.",
        "- Determine affected_queries and unchanged_queries from the edit target and the rules.",
        "- Do not include variable names or support names as query ids.",
    ]


def _audit_stress_footer(profile: str) -> list[str]:
    if profile != "strict":
        return []
    return [
        "",
        "STRICT AUDIT STRESS RULES:",
        "- Return one valid JSON object and nothing else.",
        "- inspect must be the single variable whose observation is needed, or \"none\".",
        "- audit_links must contain only exact unverifiable relation links using variable names.",
        "- Use link direction exactly as listed in the rules.",
        "- If no relation link is unverifiable, audit_links must be an empty list.",
    ]


def _stress_case(
    *,
    seed: int,
    variant: str,
    profile: str,
    prompt_lines: list[str],
    expected: dict[str, Any],
    world: SymbolWorld,
    missing_order: list[str],
) -> DiagnosticCase:
    return DiagnosticCase(
        case_id=f"seed{seed}_budgeted_inspect_{variant}",
        gate="budgeted_inspect",
        prompt="\n".join([_profile_contract(profile), *prompt_lines, *_stress_footer(profile)]),
        expected=expected,
        metadata={
            "world": world,
            "missing_order": missing_order,
            "variant": variant,
            "case_set": "budgeted_inspect_stress",
            "prompt_profile": profile,
        },
    )


def make_budgeted_inspect_stress_cases(seed: int = 0, profile: str = "strict") -> list[DiagnosticCase]:
    world = make_symbol_world(seed)
    link_2 = f"{world.mediator} -> {world.outcome}"
    answer_a = f"{world.outcome}={world.outcome_a}"
    mediator_expected = {
        "inspect": world.mediator,
        "uncertain": True,
        "audit_links": [link_2],
    }
    verified_expected = {
        "answer": answer_a,
        "answers": [answer_a, world.outcome_a],
        "inspect": "none",
        "uncertain": False,
    }

    return [
        _stress_case(
            seed=seed,
            variant="noise_first",
            profile=profile,
            world=world,
            missing_order=[world.noise, world.mediator],
            expected=mediator_expected,
            prompt_lines=[
                "",
                "Budgeted inspection stress task. Budget allows exactly one inspection.",
                "Use only the listed random-symbol rules.",
                f"Rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Observed state lists missing variables in this order: {world.noise}=unknown, {world.mediator}=unknown.",
                f"Candidates: {world.noise}, {world.mediator}, {world.outcome}, none.",
                f"Choose the single variable to inspect to verify {world.outcome}.",
            ],
        ),
        _stress_case(
            seed=seed,
            variant="mediator_first",
            profile=profile,
            world=world,
            missing_order=[world.mediator, world.noise],
            expected=mediator_expected,
            prompt_lines=[
                "",
                "Budgeted inspection stress task. Budget allows exactly one inspection.",
                "Use only the listed random-symbol rules.",
                f"Rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Observed state lists missing variables in this order: {world.mediator}=unknown, {world.noise}=unknown.",
                f"Candidates: {world.mediator}, {world.noise}, {world.outcome}, none.",
                f"Choose the single variable to inspect to verify {world.outcome}.",
            ],
        ),
        _stress_case(
            seed=seed,
            variant="many_distractors",
            profile=profile,
            world=world,
            missing_order=[world.unrelated_in, world.noise, world.unrelated_out, world.mediator],
            expected=mediator_expected,
            prompt_lines=[
                "",
                "Budgeted inspection stress task. Budget allows exactly one inspection.",
                "Use only the listed random-symbol rules.",
                f"Target rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Distractor rule: {world.unrelated_in}={world.unrelated_value} -> {world.unrelated_out}={world.unrelated_result}.",
                f"Observed state lists missing variables in this order: {world.unrelated_in}=unknown, {world.noise}=unknown, {world.unrelated_out}=unknown, {world.mediator}=unknown.",
                f"Candidates: {world.unrelated_in}, {world.noise}, {world.unrelated_out}, {world.mediator}, {world.outcome}, none.",
                f"Choose the single variable to inspect to verify {world.outcome}.",
            ],
        ),
        _stress_case(
            seed=seed,
            variant="already_verifiable",
            profile=profile,
            world=world,
            missing_order=[world.noise],
            expected=verified_expected,
            prompt_lines=[
                "",
                "Budgeted inspection stress task. Budget allows at most one inspection.",
                "Use only the listed random-symbol rules.",
                f"Rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Observed state: {world.cause}={world.cause_value}, {world.mediator}={world.mediator_a}, {world.outcome}={world.outcome_a}, {world.noise}=unknown.",
                f"Candidates: {world.noise}, {world.mediator}, {world.outcome}, none.",
                f"Choose the single variable to inspect, if any, before answering {world.outcome}.",
            ],
        ),
        _stress_case(
            seed=seed,
            variant="support_conditioned_noise_first",
            profile=profile,
            world=world,
            missing_order=[world.noise, world.mediator],
            expected=mediator_expected,
            prompt_lines=[
                "",
                "Budgeted inspection stress task. Budget allows exactly one inspection.",
                "The same symbols have different values under different supports.",
                f"Support {world.support_a}: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Support {world.support_b}: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_b} -> {world.outcome}={world.outcome_b}.",
                f"Query support: {world.support_a}. Observed state lists missing variables in this order: {world.noise}=unknown, {world.mediator}=unknown.",
                f"Candidates: {world.noise}, {world.mediator}, {world.outcome}, none.",
                f"Choose the single variable to inspect to verify {world.outcome} under support {world.support_a}.",
            ],
        ),
    ]


def make_local_edit_behavior_stress_cases(seed: int = 0, profile: str = "strict") -> list[DiagnosticCase]:
    world = make_symbol_world(seed)
    edited_target_answer = f"{world.outcome}={world.unrelated_result}"
    other_support_answer = f"{world.outcome}={world.outcome_b}"
    unrelated_answer = f"{world.unrelated_out}={world.unrelated_result}"
    prompt = "\n".join(
        [
            _profile_contract(profile),
            "",
            "Local edit behavior stress task. Apply the edit, then answer each query id.",
            "Only the edited support-specific relation should change behavior.",
            "",
            "Original rules:",
            f"- target_chain uses support {world.support_a}: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}",
            f"- other_support uses support {world.support_b}: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_b} -> {world.outcome}={world.outcome_b}",
            f"- unrelated_chain: {world.unrelated_in}={world.unrelated_value} -> {world.unrelated_out}={world.unrelated_result}",
            "",
            f"Edit exactly this relation in support {world.support_a}: {world.mediator}={world.mediator_a} -> {world.outcome}={world.unrelated_result}.",
            "",
            "Queries after the edit:",
            f"- target_chain: support={world.support_a}, observed {world.cause}={world.cause_value}.",
            f"- other_support: support={world.support_b}, observed {world.cause}={world.cause_value}.",
            f"- unrelated_chain: observed {world.unrelated_in}={world.unrelated_value}.",
            "",
            "Return affected_queries, unchanged_queries, and answers_by_query for all three query ids.",
            *_local_edit_behavior_footer(profile),
        ]
    )
    return [
        DiagnosticCase(
            case_id=f"seed{seed}_local_edit_behavior",
            gate="local_edit_locality",
            prompt=prompt,
            expected={
                "affected_queries": ["target_chain"],
                "unchanged_queries": ["other_support", "unrelated_chain"],
                "answers_by_query": {
                    "target_chain": edited_target_answer,
                    "other_support": other_support_answer,
                    "unrelated_chain": unrelated_answer,
                },
            },
            metadata={
                "world": world,
                "case_set": "local_edit_behavior_stress",
                "prompt_profile": profile,
            },
        )
    ]


def _audit_case(
    *,
    seed: int,
    variant: str,
    profile: str,
    prompt_lines: list[str],
    expected: dict[str, Any],
    world: SymbolWorld,
    missing_order: list[str],
) -> DiagnosticCase:
    return DiagnosticCase(
        case_id=f"seed{seed}_audit_{variant}",
        gate="audit_correctness",
        prompt="\n".join([_profile_contract(profile), *prompt_lines, *_audit_stress_footer(profile)]),
        expected={**expected, "exact_audit_links": True},
        metadata={
            "world": world,
            "missing_order": missing_order,
            "variant": variant,
            "case_set": "audit_correctness_stress",
            "prompt_profile": profile,
        },
    )


def make_audit_correctness_stress_cases(seed: int = 0, profile: str = "strict") -> list[DiagnosticCase]:
    world = make_symbol_world(seed)
    link_1 = f"{world.cause} -> {world.mediator}"
    link_2 = f"{world.mediator} -> {world.outcome}"
    unrelated_link = f"{world.unrelated_in} -> {world.unrelated_out}"

    return [
        _audit_case(
            seed=seed,
            variant="mediator_missing",
            profile=profile,
            world=world,
            missing_order=[world.mediator],
            expected={"inspect": world.mediator, "uncertain": True, "audit_links": [link_2]},
            prompt_lines=[
                "",
                "Audit correctness stress task. Name the exact unverifiable relation link.",
                f"Rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Observed state: {world.cause}={world.cause_value}, {world.mediator}=unknown.",
                f"Can {world.outcome} be verified without inspection?",
            ],
        ),
        _audit_case(
            seed=seed,
            variant="cause_missing",
            profile=profile,
            world=world,
            missing_order=[world.cause],
            expected={"inspect": world.cause, "uncertain": True, "audit_links": [link_1]},
            prompt_lines=[
                "",
                "Audit correctness stress task. Name the first exact unverifiable relation link in the chain.",
                f"Rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Observed state: {world.cause}=unknown, {world.mediator}=unknown.",
                f"What must be inspected first to verify {world.outcome}, and which relation link is unverifiable first?",
            ],
        ),
        _audit_case(
            seed=seed,
            variant="support_distractor",
            profile=profile,
            world=world,
            missing_order=[world.mediator],
            expected={"inspect": world.mediator, "uncertain": True, "audit_links": [link_2]},
            prompt_lines=[
                "",
                "Audit correctness stress task. Use only the query support and ignore distractor support rules.",
                f"Support {world.support_a}: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Support {world.support_b}: {world.cause}={world.cause_value} -> {world.unrelated_in}={world.unrelated_value} -> {world.outcome}={world.outcome_b}.",
                f"Query support: {world.support_a}. Observed state: {world.cause}={world.cause_value}, {world.mediator}=unknown.",
                f"Which exact relation link blocks verification of {world.outcome}?",
            ],
        ),
        _audit_case(
            seed=seed,
            variant="outcome_observation_missing",
            profile=profile,
            world=world,
            missing_order=[world.outcome],
            expected={"inspect": world.outcome, "uncertain": True, "audit_links": [link_2]},
            prompt_lines=[
                "",
                "Audit correctness stress task. The predicted chain value is known, but the outcome observation is missing.",
                f"Rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Observed state: {world.cause}={world.cause_value}, {world.mediator}={world.mediator_a}, {world.outcome}=unknown.",
                f"Which variable must be inspected to verify the observed {world.outcome}, and which relation link is being checked?",
            ],
        ),
        _audit_case(
            seed=seed,
            variant="irrelevant_missing",
            profile=profile,
            world=world,
            missing_order=[world.noise],
            expected={"inspect": "none", "uncertain": False, "audit_links": []},
            prompt_lines=[
                "",
                "Audit correctness stress task. Do not audit irrelevant missing variables.",
                f"Target rules: {world.cause}={world.cause_value} -> {world.mediator}={world.mediator_a} -> {world.outcome}={world.outcome_a}.",
                f"Distractor rule: {world.unrelated_in}={world.unrelated_value} -> {world.unrelated_out}={world.unrelated_result}.",
                f"Observed state: {world.cause}={world.cause_value}, {world.mediator}={world.mediator_a}, {world.outcome}={world.outcome_a}, {world.noise}=unknown.",
                f"Can {world.outcome} be verified, and which relation link is unverifiable?",
            ],
        ),
    ]


def make_case_set(seed: int = 0, profile: str = "base", case_set: str = "core") -> list[DiagnosticCase]:
    if case_set == "core":
        return make_cases(seed=seed, profile=profile)
    if case_set == "budgeted_inspect_stress":
        return make_budgeted_inspect_stress_cases(seed=seed, profile=profile)
    if case_set == "local_edit_behavior_stress":
        return make_local_edit_behavior_stress_cases(seed=seed, profile=profile)
    if case_set == "audit_correctness_stress":
        return make_audit_correctness_stress_cases(seed=seed, profile=profile)
    raise ValueError(f"unknown case_set: {case_set}")
