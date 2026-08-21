"""Portable plain-text contracts shared by generation, judging, and review."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

from slm_synth.telemetry import aggregate_llm_telemetry

T = TypeVar("T")


class PlainTextBackend(Protocol):
    def generate_text_with_metadata(
        self, *, prompt: str, system_prompt: str = "Follow the instructions exactly."
    ) -> dict[str, Any]: ...


class PlainOutputContractError(ValueError):
    """Raised after text was returned but remained unparsable after retries."""

    def __init__(
        self,
        message: str,
        *,
        response: str,
        telemetry: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.response = response
        self.telemetry = dict(telemetry or {})


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object, tolerating only a surrounding Markdown fence."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("model returned empty content")
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return one valid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("model response must be one JSON object")
    return value


@dataclass(frozen=True)
class JudgeDecision:
    assessable: bool
    accepted: bool
    reason: str


@dataclass(frozen=True)
class ReviewDecision:
    agreed: bool
    reason: str


def _parse_labels(text: str, required: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        label, separator, value = raw_line.partition(":")
        normalized = label.strip().upper()
        if separator and normalized in required and normalized not in values:
            values[normalized] = value.strip()
    missing = [label for label in required if not values.get(label)]
    if missing:
        raise ValueError(f"model decision missing labeled field(s): {missing}")
    return values


def parse_judge_decision(text: str) -> JudgeDecision:
    values = _parse_labels(text, ("ASSESSABLE", "DECISION", "REASON"))
    if values["ASSESSABLE"].upper() not in {"YES", "NO"}:
        raise ValueError("ASSESSABLE must be YES or NO")
    if values["DECISION"].upper() not in {"ACCEPT", "REJECT"}:
        raise ValueError("DECISION must be ACCEPT or REJECT")
    assessable = values["ASSESSABLE"].upper() == "YES"
    accepted = assessable and values["DECISION"].upper() == "ACCEPT"
    return JudgeDecision(assessable=assessable, accepted=accepted, reason=values["REASON"])


def parse_review_decision(text: str) -> ReviewDecision:
    values = _parse_labels(text, ("AGREE", "REASON"))
    if values["AGREE"].upper() not in {"YES", "NO"}:
        raise ValueError("AGREE must be YES or NO")
    return ReviewDecision(agreed=values["AGREE"].upper() == "YES", reason=values["REASON"])


def call_plain_text(
    backend: Any, *, prompt: str, system_prompt: str
) -> tuple[str, dict[str, Any]]:
    """Call the portable contract and normalize its telemetry envelope."""
    method = getattr(backend, "generate_text_with_metadata", None)
    if method is None:
        raise TypeError("backend does not implement the portable plain-text contract")
    result = method(prompt=prompt, system_prompt=system_prompt)
    if not isinstance(result, Mapping):
        raise TypeError("plain-text backend returned a non-object envelope")
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("plain-text backend returned empty content")
    telemetry = result.get("telemetry")
    return text.strip(), dict(telemetry) if isinstance(telemetry, Mapping) else {}


def call_plain_parsed(
    backend: Any,
    *,
    prompt: str,
    system_prompt: str,
    parser: Callable[[str], T],
    attempts: int = 3,
) -> tuple[T, dict[str, Any]]:
    """Retry only malformed role output; semantic decisions are never retried."""
    telemetry: list[dict[str, Any]] = []
    last_error: ValueError | None = None
    last_response = ""
    active_prompt = prompt
    for _attempt in range(1, attempts + 1):
        text, call_telemetry = call_plain_text(
            backend, prompt=active_prompt, system_prompt=system_prompt
        )
        last_response = text
        telemetry.append(call_telemetry)
        try:
            return parser(text), aggregate_llm_telemetry(telemetry)
        except ValueError as exc:
            last_error = exc
            active_prompt = (
                prompt
                + "\n\nYour prior response violated the exact output contract. "
                "Return only the requested fields."
            )
    raise PlainOutputContractError(
        f"model output remained malformed after {attempts} attempts: {last_error}",
        response=last_response,
        telemetry=aggregate_llm_telemetry(telemetry),
    )
