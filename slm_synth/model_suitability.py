"""Shared model suitability policy for OpenRouter-backed repository workflows."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

OPENROUTER_MODEL_URL = "https://openrouter.ai/api/v1/model/{author}/{slug}"


class ModelSuitabilityError(RuntimeError):
    """Selected model cannot satisfy repository-wide runtime policy."""


@dataclass(frozen=True)
class ReasoningSuitability:
    model: str
    reasoning_capable: bool
    reasoning_mandatory: bool
    reasoning_disable_supported: bool
    reasoning_policy_pass: bool
    source: str = "OpenRouter"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=256)
def _load_openrouter_model(model: str) -> dict[str, Any]:
    try:
        author, slug = model.split("/", 1)
    except ValueError as exc:
        raise ModelSuitabilityError(
            f"OpenRouter model ID must be author/slug, got {model!r}"
        ) from exc
    url = OPENROUTER_MODEL_URL.format(
        author=quote(author, safe=""),
        slug=quote(slug, safe=":"),
    )
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:  # network/provider metadata failure is fail-closed
        raise ModelSuitabilityError(
            f"Unable to verify OpenRouter reasoning policy for {model!r}: {exc}"
        ) from exc
    metadata = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise ModelSuitabilityError("OpenRouter model metadata returned an invalid payload")
    return metadata


@lru_cache(maxsize=256)
def get_reasoning_suitability(model: str) -> ReasoningSuitability:
    """Return fail-closed reasoning suitability for one OpenRouter model."""
    model_id = str(model).strip()
    metadata = _load_openrouter_model(model_id)
    reasoning = metadata.get("reasoning")
    supported_parameters = metadata.get("supported_parameters")
    supports_reasoning_parameter = (
        isinstance(supported_parameters, list) and "reasoning" in supported_parameters
    )
    reasoning_capable = isinstance(reasoning, dict) or supports_reasoning_parameter

    if not reasoning_capable:
        return ReasoningSuitability(
            model=model_id,
            reasoning_capable=False,
            reasoning_mandatory=False,
            reasoning_disable_supported=True,
            reasoning_policy_pass=True,
        )

    reasoning_metadata = reasoning if isinstance(reasoning, dict) else {}
    mandatory = bool(reasoning_metadata.get("mandatory", False))
    efforts_exposed = "supported_efforts" in reasoning_metadata
    supported_efforts = reasoning_metadata.get("supported_efforts")
    disable_supported = (
        not mandatory
        and efforts_exposed
        and (
            supported_efforts is None
            or (isinstance(supported_efforts, list) and "none" in supported_efforts)
        )
    )

    return ReasoningSuitability(
        model=model_id,
        reasoning_capable=True,
        reasoning_mandatory=mandatory,
        reasoning_disable_supported=disable_supported,
        reasoning_policy_pass=disable_supported,
    )


def require_reasoning_off_suitability(model: str) -> ReasoningSuitability:
    suitability = get_reasoning_suitability(model)
    if not suitability.reasoning_policy_pass:
        raise ModelSuitabilityError(
            f"Model {model!r} is ineligible: repository policy requires verified reasoning.effort=none support"
        )
    return suitability
