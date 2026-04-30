from __future__ import annotations

import json
import re
from typing import Any

from .cases import DiagnosticCase, SymbolWorld


class SolverError(RuntimeError):
    pass


def parse_json_response(text: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return {"raw": text, "parse_error": "no_json_object"}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except json.JSONDecodeError as exc:
            return {"raw": text, "parse_error": str(exc)}


class DiagnosticSolver:
    name = "diagnostic_solver"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        raise NotImplementedError


def _expected_answer(case: DiagnosticCase) -> str:
    answers = case.expected.get("answers", [])
    return answers[0] if answers else ""


def _world(case: DiagnosticCase) -> SymbolWorld:
    world = case.metadata.get("world")
    if not isinstance(world, SymbolWorld):
        raise SolverError("case has no SymbolWorld metadata")
    return world


class RelationOracleSolver(DiagnosticSolver):
    name = "relation_oracle"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        response: dict[str, Any] = {}
        if "answers" in case.expected:
            response["answer"] = _expected_answer(case)
        if "inspect" in case.expected:
            response["inspect"] = case.expected["inspect"]
        if "uncertain" in case.expected:
            response["uncertain"] = case.expected["uncertain"]
        if "audit_links" in case.expected:
            response["audit_links"] = list(case.expected["audit_links"])
        if "affected_queries" in case.expected:
            response["affected_queries"] = list(case.expected["affected_queries"])
        if "unchanged_queries" in case.expected:
            response["unchanged_queries"] = list(case.expected["unchanged_queries"])
        if "answers_by_query" in case.expected:
            response["answers_by_query"] = dict(case.expected["answers_by_query"])
        return response


class GlobalMappingSolver(DiagnosticSolver):
    name = "global_mapping"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        response: dict[str, Any] = {}
        if "answers" in case.expected:
            response["answer"] = case.metadata.get("global_answer", _expected_answer(case))
        if "inspect" in case.expected:
            response["inspect"] = case.expected["inspect"]
        if "uncertain" in case.expected:
            response["uncertain"] = case.expected["uncertain"]
        if "audit_links" in case.expected:
            response["audit_links"] = list(case.expected["audit_links"])
        if "affected_queries" in case.expected:
            response["affected_queries"] = list(case.expected["affected_queries"])
        if "unchanged_queries" in case.expected:
            response["unchanged_queries"] = list(case.expected["unchanged_queries"])
        return response


class EditComplianceSolver(DiagnosticSolver):
    name = "edit_compliance"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        response = RelationOracleSolver().solve(case)
        if case.gate == "local_edit_locality":
            response["affected_queries"] = ["target_chain", "other_support", "unrelated_chain"]
            response["unchanged_queries"] = []
        return response


class EditNoBehaviorSolver(DiagnosticSolver):
    name = "edit_no_behavior"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        response = RelationOracleSolver().solve(case)
        if case.gate == "local_edit_locality" and "answers_by_query" in response:
            world = _world(case)
            answers = dict(response["answers_by_query"])
            answers["target_chain"] = f"{world.outcome}={world.outcome_a}"
            response["answers_by_query"] = answers
        return response


class MissingnessTemplateSolver(DiagnosticSolver):
    name = "missingness_template"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        response = RelationOracleSolver().solve(case)
        missing_order = case.metadata.get("missing_order", [])
        if missing_order:
            response["uncertain"] = True
            response["inspect"] = missing_order[0]
            world = _world(case)
            response["audit_links"] = [f"{world.cause} -> {world.mediator}", f"{world.mediator} -> {world.outcome}"]
        return response


class SurfaceAuditSolver(DiagnosticSolver):
    name = "surface_audit"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        response = RelationOracleSolver().solve(case)
        if "audit_links" in case.expected:
            response["audit_links"] = ["some relation is uncertain"]
        return response


class ReverseAuditSolver(DiagnosticSolver):
    name = "reverse_audit"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        response = RelationOracleSolver().solve(case)
        if "audit_links" in case.expected:
            reversed_links = []
            for link in case.expected["audit_links"]:
                parts = [part.strip() for part in str(link).split("->")]
                if len(parts) == 2:
                    reversed_links.append(f"{parts[1]} -> {parts[0]}")
                else:
                    reversed_links.append(str(link))
            response["audit_links"] = reversed_links
        return response


class OutcomeAuditSolver(DiagnosticSolver):
    name = "outcome_audit"

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        response = RelationOracleSolver().solve(case)
        if "inspect" in case.expected and case.expected["inspect"] != "none":
            world = _world(case)
            response["inspect"] = world.outcome
            response["audit_links"] = [f"{world.outcome} -> {world.outcome}"]
        return response


class LlamaCppJsonSolver(DiagnosticSolver):
    name = "llama_cpp"

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def solve(self, case: DiagnosticCase) -> dict[str, Any]:
        import requests

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": case.prompt}],
                "temperature": 0.0,
                "max_tokens": 256,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        parsed["_raw_text"] = content
        parsed["_usage"] = data.get("usage", {})
        return parsed


def make_solver(name: str, base_url: str = "http://127.0.0.1:8083", model: str = "local") -> DiagnosticSolver:
    if name == "relation_oracle":
        return RelationOracleSolver()
    if name == "global_mapping":
        return GlobalMappingSolver()
    if name == "edit_compliance":
        return EditComplianceSolver()
    if name == "edit_no_behavior":
        return EditNoBehaviorSolver()
    if name == "missingness_template":
        return MissingnessTemplateSolver()
    if name == "surface_audit":
        return SurfaceAuditSolver()
    if name == "reverse_audit":
        return ReverseAuditSolver()
    if name == "outcome_audit":
        return OutcomeAuditSolver()
    if name == "llama_cpp":
        return LlamaCppJsonSolver(base_url=base_url, model=model)
    raise ValueError(f"unknown solver: {name}")
