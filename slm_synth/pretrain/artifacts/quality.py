from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from typing import Any

from slm_synth.pretrain.artifacts.base import GroundedArtifact

FORBIDDEN_PLACEHOLDERS = (
    "lalala", "lala lala", "example person", "placeholder", "test person", "company x",
)


def _safe_eval(expression: str) -> int | None:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and right:
                return left / right
        raise ValueError("unsupported expression")

    try:
        result = visit(tree)
    except ValueError:
        return None
    return int(result) if result.denominator == 1 else None


def artifact_fingerprint(artifact: GroundedArtifact) -> str:
    canonical = json.dumps({"signal": artifact.signal, "family": artifact.family, "payload": artifact.payload}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def artifact_structure_fingerprint(artifact: GroundedArtifact) -> str:
    canonical = json.dumps(artifact.payload, sort_keys=True, ensure_ascii=False)
    if artifact.signal == "task_code":
        canonical = re.sub(r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", "def FUNCTION(", canonical)
    canonical = re.sub(r"\b\d+\b", "N", canonical)
    return hashlib.sha256(f"{artifact.signal}:{artifact.family}:{canonical}".encode("utf-8")).hexdigest()


def validate_artifact(artifact: GroundedArtifact) -> list[str]:
    payload = artifact.payload
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    issues = [f"placeholder_text:{token}" for token in FORBIDDEN_PLACEHOLDERS if token in serialized]

    if artifact.signal in {"arithmetic", "educational_qa_mcq_math"}:
        computed = _safe_eval(str(payload.get("expression", "")))
        try:
            answer = int(str(payload.get("answer", "")))
        except ValueError:
            answer = None
        if computed is None or answer != computed:
            issues.append("invalid_grounded_math_answer")

    if artifact.signal == "educational_qa_mcq_math":
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 4 or len(set(choices)) != 4:
            issues.append("invalid_choices")
        elif str(payload.get("answer")) not in choices:
            issues.append("answer_not_in_choices")
        elif artifact.family in {"missing_operand", "exact_division", "two_step_quantity"}:
            if any(re.fullmatch(r"-\d+", choice.strip()) for choice in choices):
                issues.append("implausible_negative_distractor")

    if artifact.signal == "educational_qa_mcq_general":
        choices = payload.get("choices")
        answer = payload.get("answer")
        if not isinstance(choices, list) or len(choices) != 4 or len(set(choices)) != 4:
            issues.append("invalid_choices")
        elif answer not in choices:
            issues.append("answer_not_in_choices")
        if not str(payload.get("evidence", "")).strip():
            issues.append("missing_evidence")

    if artifact.signal == "task_code":
        try:
            tree = ast.parse(str(payload.get("code", "")))
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                issues.append("not_single_function")
        except SyntaxError:
            issues.append("invalid_python_code")

    if artifact.signal == "factual_restraint":
        question = str(payload.get("question", ""))
        behavior = str(payload.get("behavior", ""))
        if not question or not behavior:
            issues.append("incomplete_restraint_artifact")
        if re.search(r"\bmonth\s+\d+\b", question, re.IGNORECASE):
            issues.append("unnatural_month_placeholder")

    return issues


def assert_valid_artifacts(artifacts: list[GroundedArtifact]) -> None:
    ids = [artifact.artifact_id for artifact in artifacts]
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    errors: list[str] = [f"duplicate_artifact_id:{key}" for key in duplicate_ids]
    for artifact in artifacts:
        errors.extend(f"{artifact.artifact_id}:{issue}" for issue in validate_artifact(artifact))
    if errors:
        raise ValueError("Grounded artifact preflight failed: " + "; ".join(errors))
