"""Shared Hugging Face dataset push helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
