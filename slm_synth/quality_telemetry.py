"""Telemetry aggregation for multi-role quality pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from slm_synth.telemetry import aggregate_llm_telemetry


def combine_telemetry(*items: Mapping[str, Any]) -> dict[str, Any]:
    return aggregate_llm_telemetry([dict(item) for item in items if item])
