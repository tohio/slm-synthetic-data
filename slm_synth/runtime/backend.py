from __future__ import annotations

from typing import Any

# Keep the proven production backend construction path used by the finalized
# one-offs. Dataset migration can therefore share runtime code without changing
# provider, routing, retry, structured-output, telemetry, or adaptive-admission
# behavior.
from slm_synth.sft.generation import build_openrouter_backend


def build_backend(
    *,
    model: str,
    max_tokens: int,
    concurrency: int,
    routing_mode: str,
    provider: str | None,
    temperature: float | None,
    top_p: float | None,
) -> Any:
    """Build the OpenRouter backend exactly as the finalized one-offs do."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    return build_openrouter_backend(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        adaptive_maximum_in_flight=concurrency,
        adaptive_initial_in_flight=concurrency,
        openrouter_routing_mode=routing_mode,
        openrouter_provider=provider,
    )
