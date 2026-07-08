"""Dataset-card generation helpers for response-distillation runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    """Load a run-level distillation manifest from JSON."""
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("run manifest must contain a JSON object")
    return dict(value)


def render_dataset_card(
    *,
    run_manifest: Mapping[str, Any],
    dataset_name: str,
    license_name: str | None = None,
    language: str = "en",
) -> str:
    """Render a concise dataset card from a local run manifest.

    Teacher/provider/run provenance belongs here, not in public JSONL rows.
    """
    manifest = _validate_run_manifest(run_manifest)
    clean_dataset_name = _require_non_empty_string(dataset_name, "dataset_name")
    clean_language = _require_non_empty_string(language, "language")
    clean_license = license_name.strip() if isinstance(license_name, str) and license_name.strip() else None

    front_matter = [
        "---",
        f'language: "{clean_language}"',
    ]
    if clean_license is not None:
        front_matter.append(f'license: "{clean_license}"')
    front_matter.append("---")

    rows = [
        "| Signal | Rows | Dataset file |",
        "| --- | ---: | --- |",
    ]
    for dataset in manifest["datasets"]:
        rows.append(
            "| {signal} | {row_count} | `{dataset_path}` |".format(
                signal=dataset["signal"],
                row_count=dataset["row_count"],
                dataset_path=dataset["dataset_path"],
            )
        )

    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    generation_lines = [
        f"- Generation run: `{manifest['generation_run']}`",
        f"- Teacher provider: `{manifest['teacher_provider']}`",
        f"- Teacher model: `{manifest['teacher_model']}`",
    ]
    if manifest.get("token_target") is not None:
        generation_lines.append(f"- Token target: `{manifest['token_target']}`")
    if metadata.get("target_rows") is not None:
        generation_lines.append(f"- Target rows: `{metadata['target_rows']}`")
    if metadata.get("planned_prompt_rows") is not None:
        generation_lines.append(f"- Planned prompt rows: `{metadata['planned_prompt_rows']}`")
    if metadata.get("accepted_rows") is not None:
        generation_lines.append(f"- Accepted rows: `{metadata['accepted_rows']}`")
    if metadata.get("rejected_rows") is not None:
        generation_lines.append(f"- Rejected rows: `{metadata['rejected_rows']}`")
    generation_lines.append(f"- Total rows: `{manifest['total_rows']}`")

    sections = [
        "\n".join(front_matter),
        f"# {clean_dataset_name}",
        "## Summary",
        (
            "Signal-specific response-distillation datasets generated from local prompts "
            "and teacher responses. Public rows contain only `id`, `prompt`, `reasoning`, "
            "and `response`; `reasoning` is always null."
        ),
        "## Generation",
        "\n".join(generation_lines),
        "## Signals",
        "\n".join(rows),
        "## Row Schema",
        "\n".join(
            [
                "Each JSONL row uses this public schema:",
                "",
                "```json",
                '{ "id": "string", "prompt": "string", "reasoning": null, "response": "string" }',
                "```",
            ]
        ),
        "## Excluded From Rows",
        (
            "Signal names, teacher details, provider details, generation-run metadata, "
            "difficulty labels, and internal metadata are intentionally excluded from "
            "public training rows."
        ),
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def write_dataset_card(
    *,
    run_manifest_path: str | Path,
    output_path: str | Path,
    dataset_name: str,
    license_name: str | None = None,
    language: str = "en",
) -> Path:
    """Write a dataset card rendered from a run-level manifest."""
    manifest = load_run_manifest(run_manifest_path)
    text = render_dataset_card(
        run_manifest=manifest,
        dataset_name=dataset_name,
        license_name=license_name,
        language=language,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _validate_run_manifest(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(run_manifest)
    for field in (
        "generation_run",
        "teacher_model",
        "teacher_provider",
        "token_target",
        "datasets",
        "total_rows",
    ):
        if field not in manifest:
            raise ValueError(f"run manifest missing required field: {field}")

    manifest["generation_run"] = _require_non_empty_string(manifest["generation_run"], "generation_run")
    manifest["teacher_model"] = _require_non_empty_string(manifest["teacher_model"], "teacher_model")
    provider = _require_non_empty_string(manifest["teacher_provider"], "teacher_provider").lower()
    if provider != "openrouter":
        raise ValueError("teacher_provider must be 'openrouter'")
    manifest["teacher_provider"] = provider

    total_rows = manifest["total_rows"]
    if not isinstance(total_rows, int) or total_rows < 0:
        raise ValueError("total_rows must be a non-negative integer")

    datasets = manifest["datasets"]
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
        raise ValueError("datasets must be a list of objects")
    manifest["datasets"] = [_validate_dataset_entry(dataset) for dataset in datasets]
    return manifest


def _validate_dataset_entry(dataset: Any) -> dict[str, Any]:
    if not isinstance(dataset, Mapping):
        raise ValueError("each dataset entry must be an object")
    signal = _require_non_empty_string(dataset.get("signal"), "dataset signal")
    dataset_path = _require_non_empty_string(dataset.get("dataset_path"), "dataset_path")
    row_count = dataset.get("row_count")
    if not isinstance(row_count, int) or row_count < 0:
        raise ValueError("dataset row_count must be a non-negative integer")
    return {
        "signal": signal,
        "dataset_path": dataset_path,
        "row_count": row_count,
    }


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
