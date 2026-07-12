from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any


_FUNCTION_SIGNATURE_ONLY_RE = re.compile(
    r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*:?\s*$"
)
_FUNCTION_DEFINITION_RE = re.compile(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", re.MULTILINE)


def validate_code_generation_pair_quality(row: Mapping[str, Any]) -> list[str]:
    """Return code-generation-specific rejection reasons for a distillation-DPO row."""
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or not _is_code_generation_row(metadata):
        return []

    chosen_text = _messages_to_text(row.get("chosen"))
    if not _looks_like_python_code(chosen_text):
        return []

    reasons: list[str] = []
    if _is_signature_only(chosen_text):
        reasons.append("signature_only_chosen_code")
        return reasons

    if _FUNCTION_DEFINITION_RE.search(chosen_text):
        try:
            ast.parse(chosen_text)
        except SyntaxError:
            reasons.append("invalid_chosen_python_syntax")
        else:
            if _has_empty_function_body(chosen_text):
                reasons.append("empty_chosen_function_body")

    return reasons


def _is_code_generation_row(metadata: Mapping[str, Any]) -> bool:
    values = {
        str(metadata.get("category", "")),
        str(metadata.get("template_family", "")),
        str(metadata.get("eval_family", "")),
        str(metadata.get("failure_mode", "")),
    }
    text = " ".join(values).lower()
    return (
        "code_generation" in text
        or "code generation" in text
        or "code_function" in text
        or "function_body" in text
        or "signature" in text
    )


def _messages_to_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, Sequence):
        return ""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts).strip()


def _looks_like_python_code(text: str) -> bool:
    stripped = text.strip()
    return bool(
        _FUNCTION_DEFINITION_RE.search(stripped)
        or stripped.startswith(("return ", "if ", "for ", "while ", "class "))
        or "\n    " in stripped
    )


def _is_signature_only(text: str) -> bool:
    return bool(_FUNCTION_SIGNATURE_ONLY_RE.match(text.strip()))


def _has_empty_function_body(text: str) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.body:
                return True
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                return True
            if len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return True
    return False
