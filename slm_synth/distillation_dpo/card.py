"""Dataset card helpers for distillation-DPO artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from slm_synth.distillation_dpo.io import CHOSEN_SOURCE, DATASET_TYPE, REJECTED_SOURCE, TARGET_CONSUMER
from slm_synth.distillation_dpo.seeds import validate_family


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid run manifest JSON: {manifest_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("run manifest must be a JSON object")
    return _validate_run_manifest(value)


def render_dataset_card(
    *,
    run_manifest: Mapping[str, Any],
    dataset_name: str,
    license_name: str | None = None,
    language: str = "en",
) -> str:
    manifest = _validate_run_manifest(run_manifest)
    clean_dataset_name = _require_non_empty_string(dataset_name, "dataset_name")
    clean_language = _require_non_empty_string(language, "language")

    front_matter = ["---", f"language: {clean_language}"]
    if license_name:
        front_matter.append(f"license: {license_name.strip()}")
    front_matter.append("---")

    rows = []
    for dataset in manifest["datasets"]:
        rows.append(
            f"- `{dataset['family']}`: `{dataset['row_count']}` rows, `{dataset['dataset_path']}`"
        )

    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    planning_lines = _planning_lines(metadata)

    sections = [
        "\n".join(front_matter),
        f"# {clean_dataset_name}",
        "## Summary",
        (
            "Distillation-DPO preference datasets for aligning distilled SLM outputs. "
            "Public rows contain prompt, chosen, rejected, and metadata fields only."
        ),
        "## Lineage",
        "\n".join(
            [
                f"- Dataset type: `{manifest['dataset_type']}`",
                f"- Generation run: `{manifest['generation_run']}`",
                f"- Teacher provider: `{manifest['teacher_provider']}`",
                f"- Teacher model: `{manifest['teacher_model']}`",
                f"- Chosen source: `{manifest['chosen_source']}`",
                f"- Rejected source: `{manifest['rejected_source']}`",
                f"- Target consumer: `{manifest['target_consumer']}`",
                f"- Total rows: `{manifest['total_rows']}`",
            ]
        ),
        "## Pair Planning",
        "\n".join(planning_lines),
        "## Families",
        "\n".join(rows),
        "## Row Schema",
        "\n".join(
            [
                "Each JSONL row uses this public schema:",
                "",
                "```json",
                '{ "id": "string", "prompt": [{"role":"user","content":"string"}], "chosen": [{"role":"assistant","content":"string"}], "rejected": [{"role":"assistant","content":"string"}], "metadata": {"category":"string","difficulty":1,"template_family":"string","eval_family":"string|null","failure_mode":"string"} }',
                "```",
            ]
        ),
        "## Excluded From Rows",
        (
            "Teacher details, provider details, generation-run metadata, source labels, "
            "cost, retry counts, and other operational metadata are intentionally "
            "excluded from public training rows."
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


def _planning_lines(metadata: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, label in (
        ("generation_mode", "Generation mode"),
        ("target_pairs", "Target pairs"),
        ("planned_pairs", "Planned pairs"),
        ("accepted_pairs", "Accepted pairs"),
        ("rejected_pairs", "Rejected pairs"),
    ):
        if metadata.get(key) is not None:
            lines.append(f"- {label}: `{metadata[key]}`")
    if not lines:
        lines.append("- Pair planning metadata was not provided in the run manifest.")
    return lines


def _validate_run_manifest(run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(run_manifest)
    required = {
        "dataset_type",
        "generation_run",
        "teacher_model",
        "teacher_provider",
        "chosen_source",
        "rejected_source",
        "target_consumer",
        "datasets",
        "total_rows",
    }
    missing = sorted(field for field in required if field not in manifest)
    if missing:
        raise ValueError(f"run manifest missing required field(s): {missing}")

    if manifest["dataset_type"] != DATASET_TYPE:
        raise ValueError(f"dataset_type must be {DATASET_TYPE!r}")
    manifest["generation_run"] = _require_non_empty_string(manifest["generation_run"], "generation_run")
    manifest["teacher_model"] = _require_non_empty_string(manifest["teacher_model"], "teacher_model")
    if _require_non_empty_string(manifest["teacher_provider"], "teacher_provider").lower() != "openrouter":
        raise ValueError("teacher_provider must be 'openrouter'")
    manifest["teacher_provider"] = "openrouter"
    if manifest["chosen_source"] != CHOSEN_SOURCE:
        raise ValueError(f"chosen_source must be {CHOSEN_SOURCE!r}")
    if manifest["rejected_source"] != REJECTED_SOURCE:
        raise ValueError(f"rejected_source must be {REJECTED_SOURCE!r}")
    if manifest["target_consumer"] != TARGET_CONSUMER:
        raise ValueError(f"target_consumer must be {TARGET_CONSUMER!r}")
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
    family = validate_family(dataset.get("family"))
    dataset_path = _require_non_empty_string(dataset.get("dataset_path"), "dataset_path")
    row_count = dataset.get("row_count")
    if not isinstance(row_count, int) or row_count < 0:
        raise ValueError("dataset row_count must be a non-negative integer")
    return {"family": family, "dataset_path": dataset_path, "row_count": row_count}


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
