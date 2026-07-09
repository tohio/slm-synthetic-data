"""Accepted-target accounting for public artifact runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COMPLETE_STATUS = "complete"
UNDERFILLED_STATUS = "underfilled"


def accepted_target_metadata(
    *,
    unit: str,
    target_count: int,
    accepted_count: int,
    attempted_count: int,
    max_backfill_rounds: int = 0,
    backfill_rounds: int = 0,
) -> dict[str, Any]:
    """Return manifest metadata for accepted-target completion status.

    ``target_count`` is the requested public artifact count. ``attempted_count`` is
    the number of candidate rows/pairs/prompts checked after quality gates. Runs
    that do not reach the accepted target are resumable, but not publish-ready.
    """
    _validate_non_negative_int(target_count, "target_count")
    _validate_non_negative_int(accepted_count, "accepted_count")
    _validate_non_negative_int(attempted_count, "attempted_count")
    _validate_non_negative_int(max_backfill_rounds, "max_backfill_rounds")
    _validate_non_negative_int(backfill_rounds, "backfill_rounds")
    if not isinstance(unit, str) or unit not in {"rows", "pairs"}:
        raise ValueError("unit must be 'rows' or 'pairs'")

    remaining = max(target_count - accepted_count, 0)
    status = COMPLETE_STATUS if remaining == 0 else UNDERFILLED_STATUS
    publish_ready = status == COMPLETE_STATUS
    budget_exhausted = remaining > 0 and backfill_rounds >= max_backfill_rounds
    payload = {
        "generation_status": status,
        "publish_ready": publish_ready,
        f"remaining_{unit}": remaining,
        "accepted_target": {
            "unit": unit,
            "target": target_count,
            "accepted": accepted_count,
            "attempted": attempted_count,
            "remaining": remaining,
            "status": status,
            "publish_ready": publish_ready,
            "max_backfill_rounds": max_backfill_rounds,
            "backfill_rounds": backfill_rounds,
            "backfill_budget_exhausted": budget_exhausted,
        },
    }
    return payload


def require_publish_ready_manifest(manifest_path: str | Path, *, artifact_name: str) -> None:
    """Reject publishing a run manifest marked as underfilled/incomplete."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{artifact_name} manifest must contain a JSON object: {path}")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        return

    accepted_target = metadata.get("accepted_target")
    generation_status = metadata.get("generation_status")
    publish_ready = metadata.get("publish_ready")

    underfilled = False
    remaining: Any = None
    if isinstance(accepted_target, dict):
        underfilled = accepted_target.get("status") == UNDERFILLED_STATUS or accepted_target.get("publish_ready") is False
        remaining = accepted_target.get("remaining")
    if generation_status == UNDERFILLED_STATUS or publish_ready is False:
        underfilled = True

    if underfilled:
        suffix = f" remaining={remaining}" if isinstance(remaining, int) else ""
        raise ValueError(
            f"{artifact_name} run is underfilled and is not publish-ready: {path}{suffix}. "
            "Run backfill/resume before pushing."
        )


def discover_run_manifest(run_dir: str | Path, *, dataset_type: str | None = None) -> Path:
    """Return the single run-level manifest under a run directory."""
    root = Path(run_dir)
    manifest_dir = root / "manifests"
    if not manifest_dir.exists():
        raise FileNotFoundError(f"manifest directory does not exist: {manifest_dir}")

    candidates: list[Path] = []
    fallback_candidates: list[Path] = []
    for manifest_path in sorted(manifest_dir.glob("*.manifest.json")):
        if ".batch" in manifest_path.name:
            continue
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        manifest_dataset_type = value.get("dataset_type")
        if dataset_type is not None and manifest_dataset_type not in {dataset_type, None}:
            continue
        fallback_candidates.append(manifest_path)
        if isinstance(value.get("datasets"), list):
            candidates.append(manifest_path)

    candidates = candidates or fallback_candidates
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        expected = f" {dataset_type}" if dataset_type else ""
        raise FileNotFoundError(f"No{expected} run manifest found under {manifest_dir}")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"Expected one run manifest under {manifest_dir}; found {len(candidates)}: {names}")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
