"""Shared Hugging Face dataset push helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi


DATASET_CARD_YAML = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.jsonl
---

"""


def dataset_card_bytes(readme_path: str | Path | None) -> bytes:
    """Return a dataset card with explicit data_files metadata for the HF viewer."""
    body = ""
    if readme_path is not None:
        path = Path(readme_path)
        if path.is_file():
            body = path.read_text(encoding="utf-8")
    body = _strip_existing_yaml_front_matter(body).lstrip()
    if not body:
        body = "# Dataset\n"
    return (DATASET_CARD_YAML + body).encode("utf-8")


def _strip_existing_yaml_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :]


def add_file_operation(path: str | Path, *, path_in_repo: str, required: bool = False) -> CommitOperationAdd | None:
    source = Path(path)
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"required HF upload artifact is missing: {source}")
        return None
    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(source))


def legacy_metadata_delete_operations(api: HfApi, *, repo_id: str) -> list[CommitOperationDelete]:
    """Delete legacy root metadata files that can confuse the HF dataset viewer."""
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return []
    operations: list[CommitOperationDelete] = []
    for path in repo_files:
        if path == "coverage.json" or path.startswith("manifests/"):
            operations.append(CommitOperationDelete(path_in_repo=path))
    return operations


def create_dataset_commit(
    api: HfApi,
    *,
    repo_id: str,
    operations: Iterable[CommitOperationAdd | CommitOperationDelete],
    commit_message: str,
) -> None:
    ops = list(operations)
    if not ops:
        return
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=commit_message,
    )

UNDERFILLED_STATUS = "underfilled"
FAILED_STATUS = "failed"


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
    if generation_status in {UNDERFILLED_STATUS, FAILED_STATUS} or publish_ready is False or metadata.get("run_failed") is True:
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