"""Shared quality-policy summaries for generator/judge/reviewer workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

JUDGE_REJECTION_INVESTIGATION_THRESHOLD = 0.05


def summarize_judge_rejections(
    manifest_paths: Iterable[str | Path],
    *,
    threshold: float = JUDGE_REJECTION_INVESTIGATION_THRESHOLD,
) -> dict[str, Any]:
    """Summarize judge decisions without changing reviewer routing semantics."""
    decisions = 0
    rejections = 0
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        quality = manifest.get("metadata", {}).get("quality_adjudication", {})
        if not isinstance(quality, dict):
            continue
        for decision in quality.values():
            if not isinstance(decision, dict) or "judge_accepted" not in decision:
                continue
            decisions += 1
            if decision.get("judge_accepted") is False:
                rejections += 1
    rate = (rejections / decisions) if decisions else 0.0
    return {
        "judge_decisions": decisions,
        "judge_rejections": rejections,
        "judge_rejection_rate": round(rate, 6),
        "judge_rejection_investigation_threshold": threshold,
        "judge_rejection_investigation_required": decisions > 0 and rate >= threshold,
    }
