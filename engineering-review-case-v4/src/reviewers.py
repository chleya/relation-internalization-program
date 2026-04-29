from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .case_schema import observed_variables


ACTION_LINKS = {
    "improve_drainage": "Drainage[t] -> PorePressureDown[t+1]",
    "add_anchoring": "Anchoring[t] -> DisplacementDown[t+1]",
    "stop_work": "StopWork[t] -> ExposureRiskDown[t]",
    "increase_monitoring": "Monitoring[t] -> UncertaintyDown[t+1]",
}

RESPONSIBILITY_BOUNDARY = [
    "human engineer remains responsible for interpretation",
    "AI review is diagnostic only",
    "no autonomous construction approval",
]

CLAIM_BOUNDARY = [
    "not a real geotechnical safety prediction",
    "not a deployment-ready engineering review",
    "toy relation-chain diagnostic only",
]

RELEVANT_VARIABLES = {
    "rainfall",
    "drainage",
    "pore_pressure",
    "displacement",
    "crack",
    "monitoring",
    "surface_warning",
}


@dataclass
class BaseReviewer:
    name: str

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _base(self, case: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "reviewer": self.name,
            "identified_variables": [],
            "relation_chain": [],
            "action_effect_points": [],
            "uncertain_links": [],
            "verification_indicators": [],
            "failure_conditions": [],
            "takeover_conditions": [],
            "responsibility_boundary": [],
            "recommended_review_status": "revise",
            "claim_boundary": [],
        }


class GenericReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("generic_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        review = self._base(case)
        review.update(
            {
                "identified_variables": ["rainfall", "support", "monitoring"],
                "verification_indicators": ["check site conditions"],
                "failure_conditions": ["if risk increases"],
                "recommended_review_status": "approve_with_conditions",
            }
        )
        return review


class SurfaceWarningReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("surface_warning_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        warning = case.get("observed_conditions", {}).get("surface_warning") == "warning"
        review = self._base(case)
        review.update(
            {
                "identified_variables": ["surface_warning"],
                "relation_chain": ["SurfaceWarning -> Danger"] if warning else [],
                "takeover_conditions": ["surface warning present"] if warning else [],
                "recommended_review_status": "takeover_required" if warning else "approve_with_conditions",
            }
        )
        return review


class StructuralMemoryReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("structural_memory_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        review = self._base(case)
        review.update(
            {
                "identified_variables": observed_variables(case),
                "relation_chain": list(case.get("known_relation_chain", [])[:1]),
                "verification_indicators": list(case.get("observed_conditions", {}).keys())[:2],
                "recommended_review_status": case.get("expected_status", "revise"),
            }
        )
        return review


class RelationChainReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("relation_chain_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        review = self._base(case)
        review.update(
            {
                "identified_variables": relevant_observed_variables(case),
                "relation_chain": list(case.get("known_relation_chain", [])),
                "action_effect_points": action_effect_points(case),
                "verification_indicators": default_verification_indicators(case),
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": "revise",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


class UncertaintyAwareReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("uncertainty_aware_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        uncertainties = list(case.get("known_uncertainties", []))
        review = self._base(case)
        review.update(
            {
                "identified_variables": relevant_observed_variables(case),
                "relation_chain": list(case.get("known_relation_chain", [])),
                "action_effect_points": action_effect_points(case),
                "uncertain_links": uncertain_links(case),
                "verification_indicators": default_verification_indicators(case) + verification_from_uncertainty(case),
                "failure_conditions": list(case.get("unsafe_review_patterns", [])),
                "takeover_conditions": takeover_conditions(case) if uncertainties else [],
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": case.get("expected_status", "revise"),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


class SchemaTemplateReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("schema_template_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        review = self._base(case)
        review.update(
            {
                "identified_variables": ["key variables listed"],
                "relation_chain": ["A -> B", "B -> C", "C -> Risk"],
                "action_effect_points": ["action -> effect"],
                "uncertain_links": ["uncertain link"],
                "verification_indicators": ["verification indicator"],
                "failure_conditions": ["failure condition"],
                "takeover_conditions": ["takeover if uncertain"],
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": case.get("expected_status", "revise"),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


class FluentNonspecificReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("fluent_nonspecific_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        review = self._base(case)
        review.update(
            {
                "identified_variables": ["weather", "geology", "construction", "monitoring"],
                "relation_chain": [
                    "The slope may become less stable when environmental and construction factors interact."
                ],
                "action_effect_points": [
                    "Use appropriate drainage, support, and monitoring according to site conditions."
                ],
                "uncertain_links": ["Some observations may need further professional confirmation."],
                "verification_indicators": ["Check monitoring data and conduct field inspection."],
                "failure_conditions": ["If conditions worsen, update the review."],
                "takeover_conditions": ["Escalate when uncertainty is significant."],
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": "revise",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


class CaseOrderMemoryReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("case_order_memory_review")
        self._expected_order = ["case_001", "case_002", "case_003", "case_004", "case_005"]
        self._index = 0

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        expected_case_id = self._expected_order[self._index % len(self._expected_order)]
        self._index += 1
        if case["case_id"] == expected_case_id:
            return UncertaintyAwareReview().review(case)

        review = self._base(case)
        review.update(
            {
                "identified_variables": ["memorized case fields"],
                "relation_chain": ["memorized link -> memorized outcome"],
                "action_effect_points": ["memorized action -> memorized effect"],
                "uncertain_links": ["memorized uncertainty"],
                "verification_indicators": ["memorized verification"],
                "takeover_conditions": ["memorized takeover"],
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": "revise",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


class BoundaryBoilerplateReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("boundary_boilerplate_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        review = self._base(case)
        review.update(
            {
                "identified_variables": observed_variables(case),
                "verification_indicators": ["verify before use"],
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": "revise",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


class UnsafeApprovalReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("unsafe_approval_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        review = UncertaintyAwareReview().review(case)
        review["reviewer"] = self.name
        review["recommended_review_status"] = "approve"
        review["claim_boundary"] = ["AI review approves this toy case as safe to proceed"]
        return review


class IrrelevantVariableReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("irrelevant_variable_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        observed = case.get("observed_conditions", {})
        distractors = [
            key
            for key in observed
            if key in {"budget_pressure", "media_attention", "contractor_confidence", "road_noise"}
        ]
        review = self._base(case)
        review.update(
            {
                "identified_variables": distractors or observed_variables(case),
                "relation_chain": [f"{item} -> RiskUp" for item in distractors] or list(case.get("known_relation_chain", [])[:1]),
                "action_effect_points": ["contractor_confidence -> approve"],
                "uncertain_links": ["irrelevant field conflict"],
                "verification_indicators": ["verify contractor confidence"],
                "takeover_conditions": ["takeover if public concern increases"],
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": "revise",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


class HiddenApprovalEchoReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("hidden_approval_echo_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        review = UncertaintyAwareReview().review(case)
        review["reviewer"] = self.name
        notes = " ".join(map(str, case.get("adversarial_notes", [])))
        if "safe to proceed" in notes.lower() or "approved" in notes.lower():
            review["recommended_review_status"] = "approve"
            review["claim_boundary"] = [notes]
        return review


class ParaphraseFragileReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("paraphrase_fragile_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        chain = list(case.get("known_relation_chain", []))
        canonical_terms = ["Rainfall", "PorePressure", "Displacement", "CrackExpansion"]
        if all(any(term in link for link in chain) for term in canonical_terms):
            return UncertaintyAwareReview().review(case)
        review = self._base(case)
        review.update(
            {
                "identified_variables": observed_variables(case),
                "relation_chain": [],
                "action_effect_points": [],
                "uncertain_links": ["unrecognized relation wording"],
                "verification_indicators": ["manual review required"],
                "takeover_conditions": ["unknown terminology"],
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": "revise",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


class FieldOrderFragileReview(BaseReviewer):
    def __init__(self) -> None:
        super().__init__("field_order_fragile_review")

    def review(self, case: dict[str, Any]) -> dict[str, Any]:
        observed = case.get("observed_conditions", {})
        first_key = next(iter(observed), "")
        if first_key == "rainfall":
            return UncertaintyAwareReview().review(case)
        review = self._base(case)
        review.update(
            {
                "identified_variables": [first_key] if first_key else [],
                "relation_chain": [f"{first_key} -> RiskUp"] if first_key else [],
                "verification_indicators": [f"verify {first_key}"] if first_key else [],
                "responsibility_boundary": RESPONSIBILITY_BOUNDARY,
                "recommended_review_status": "revise",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        return review


def action_effect_points(case: dict[str, Any]) -> list[str]:
    points: list[str] = []
    for action in case.get("proposed_actions", []):
        link = ACTION_LINKS.get(action)
        if link:
            points.append(link)
    return points


def relevant_observed_variables(case: dict[str, Any]) -> list[str]:
    return [key for key in observed_variables(case) if key in RELEVANT_VARIABLES]


def uncertain_links(case: dict[str, Any]) -> list[str]:
    chain = case.get("known_relation_chain", [])
    uncertainties = " ".join(case.get("known_uncertainties", [])).lower()
    links: list[str] = []
    for link in chain:
        low = link.lower()
        if any(term in low for term in ["porepressure", "displacement", "crack", "monitoring", "drainage"]):
            links.append(link)
    if "surface warning" in uncertainties:
        links.append("SurfaceWarning -/-> physical relation chain")
    return links or list(chain[:1])


def default_verification_indicators(case: dict[str, Any]) -> list[str]:
    indicators: list[str] = []
    observed = case.get("observed_conditions", {})
    if "pore_pressure" in observed:
        indicators.append("verify pore_pressure observation")
    if "displacement" in observed:
        indicators.append("verify displacement trend")
    if "crack" in observed:
        indicators.append("verify crack state")
    if "monitoring" in observed:
        indicators.append("verify monitoring density")
    return indicators


def verification_from_uncertainty(case: dict[str, Any]) -> list[str]:
    return [f"resolve: {item}" for item in case.get("known_uncertainties", [])]


def takeover_conditions(case: dict[str, Any]) -> list[str]:
    conditions = []
    for item in case.get("known_uncertainties", []):
        conditions.append(f"takeover if unresolved: {item}")
    return conditions


def make_reviewer(name: str) -> BaseReviewer:
    reviewers = {
        "generic_review": GenericReview,
        "surface_warning_review": SurfaceWarningReview,
        "structural_memory_review": StructuralMemoryReview,
        "relation_chain_review": RelationChainReview,
        "uncertainty_aware_review": UncertaintyAwareReview,
        "schema_template_review": SchemaTemplateReview,
        "fluent_nonspecific_review": FluentNonspecificReview,
        "case_order_memory_review": CaseOrderMemoryReview,
        "boundary_boilerplate_review": BoundaryBoilerplateReview,
        "unsafe_approval_review": UnsafeApprovalReview,
        "irrelevant_variable_review": IrrelevantVariableReview,
        "hidden_approval_echo_review": HiddenApprovalEchoReview,
        "paraphrase_fragile_review": ParaphraseFragileReview,
        "field_order_fragile_review": FieldOrderFragileReview,
    }
    if name not in reviewers:
        raise ValueError(f"unknown reviewer: {name}")
    return reviewers[name]()
