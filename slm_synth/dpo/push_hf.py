"""Push DPO run outputs to per-family Hugging Face dataset repos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, create_repo

from slm_synth.accepted_target import discover_run_manifest, require_publish_ready_manifest
from slm_synth.hf_push import (
    add_file_operation,
    create_dataset_commit,
    dataset_card_bytes,
    legacy_metadata_delete_operations,
)
from slm_synth.dpo.schema import validate_dpo_row


INTERNAL_DATASET_DIR_NAMES = {
    "batches",
    "partial",
    "partials",
    "provider",
    "provider_internal",
    "rejected",
    "retries",
    "retry",
    "scratch",
    "tmp",
}


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Missing HF token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    return token


def discover_jsonl_files(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"DPO dataset directory does not exist: {root}")
    candidates = sorted(
        path
        for path in root.rglob("*.jsonl")
        if path.is_file() and not _is_internal_dataset_path(path.relative_to(root))
    )
    files = _prefer_final_public_files(candidates)
    if not files:
        raise FileNotFoundError(f"No DPO JSONL files found in {root}")
    return files


def _is_internal_dataset_path(relative_path: Path) -> bool:
    return any(part in INTERNAL_DATASET_DIR_NAMES for part in relative_path.parts[:-1])


def _prefer_final_public_files(paths: list[Path]) -> list[Path]:
    files_by_family: dict[str, list[Path]] = {}
    for path in paths:
        files_by_family.setdefault(family_from_dataset_path(path), []).append(path)

    files: list[Path] = []
    for family_paths in files_by_family.values():
        final_files = [path for path in family_paths if ".batch" not in path.stem]
        files.extend(final_files or family_paths)
    return sorted(files)


def count_and_validate_jsonl(path: str | Path) -> int:
    count = 0
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {jsonl_path} at line {line_number}: {exc}") from exc
            validate_dpo_row(row)
            count += 1
    return count


def family_from_dataset_path(path: str | Path) -> str:
    stem = Path(path).stem
    return stem.split(".batch", 1)[0]


def slugify_family(family: str) -> str:
    slug = family.strip().lower().replace("_", "-")
    if not slug:
        raise ValueError("family slug cannot be empty")
    return slug


def repo_id_for_family(*, repo_owner: str, repo_prefix: str, family: str) -> str:
    owner = repo_owner.strip().strip("/")
    prefix = repo_prefix.strip().strip("-")
    if not owner:
        raise ValueError("repo_owner is required")
    if not prefix:
        raise ValueError("repo_prefix is required")
    return f"{owner}/{prefix}-{slugify_family(family)}"


def _artifact_manifest_paths(*, run_dir: Path, family: str, run_manifest: Path | None, skip_manifests: bool) -> list[Path]:
    if skip_manifests:
        return []
    manifest_dir = run_dir / "manifests"
    paths: list[Path] = []
    if run_manifest is not None:
        paths.append(run_manifest)
    if manifest_dir.exists():
        for manifest_path in sorted(manifest_dir.glob("*.manifest.json")):
            if manifest_path in paths or ".batch" in manifest_path.name:
                continue
            if manifest_path.name.startswith(f"{family}."):
                paths.append(manifest_path)
    return paths


def push_dpo_run(
    *,
    dataset_dir: str | Path,
    repo_owner: str,
    repo_prefix: str = "slm-synthetic-dpo",
    private: bool = False,
    env_file: str | None = None,
    run_dir: str | Path | None = None,
    skip_manifests: bool = False,
) -> dict[str, Any]:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    token = get_hf_token()
    api = HfApi(token=token)

    dataset_root = Path(dataset_dir)
    root = Path(run_dir) if run_dir is not None else None
    run_manifest: Path | None = None
    if root is not None:
        run_manifest = discover_run_manifest(root, dataset_type="dpo")
        require_publish_ready_manifest(run_manifest, artifact_name="DPO")
    files = discover_jsonl_files(dataset_root)
    files_by_family: dict[str, list[Path]] = {}
    for file_path in files:
        files_by_family.setdefault(family_from_dataset_path(file_path), []).append(file_path)

    repos: dict[str, dict[str, Any]] = {}
    for family, family_files in sorted(files_by_family.items()):
        repo_id = repo_id_for_family(repo_owner=repo_owner, repo_prefix=repo_prefix, family=family)
        create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        total_rows = 0
        uploaded_files: list[str] = []
        operations = legacy_metadata_delete_operations(api, repo_id=repo_id)

        for file_path in family_files:
            row_count = count_and_validate_jsonl(file_path)
            total_rows += row_count
            path_in_repo = f"data/{file_path.relative_to(dataset_root).as_posix()}"
            print(f"[push_hf] staging {file_path} -> {repo_id}/{path_in_repo} rows={row_count}")
            operations.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(file_path)))
            uploaded_files.append(path_in_repo)

        if root is not None:
            readme_path = root / "README.md"
            if not readme_path.is_file():
                raise FileNotFoundError(f"required HF dataset card source is missing: {readme_path}")
            operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=dataset_card_bytes(readme_path)))
            coverage_op = add_file_operation(root / "coverage.json", path_in_repo="artifacts/coverage.json")
            if coverage_op is not None:
                operations.append(coverage_op)
            for manifest_path in _artifact_manifest_paths(
                run_dir=root,
                family=family,
                run_manifest=run_manifest,
                skip_manifests=skip_manifests,
            ):
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"artifacts/manifests/{manifest_path.name}",
                        path_or_fileobj=str(manifest_path),
                    )
                )

        print(f"[push_hf] committing {len(operations)} file operation(s) to {repo_id}")
        create_dataset_commit(
            api,
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Update DPO family dataset: {family}",
        )

        repos[family] = {"repo_id": repo_id, "files": uploaded_files, "rows": total_rows}
        print(f"[push_hf] Completed DPO push repo={repo_id} files={len(uploaded_files)} rows={total_rows}")

    result = {"repos": repos, "repo_count": len(repos), "rows": sum(item["rows"] for item in repos.values())}
    print(f"[push_hf] Completed DPO family push repos={len(repos)} rows={result['rows']}")
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Push DPO run outputs to per-family Hugging Face dataset repos.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-owner", required=True)
    parser.add_argument("--repo-prefix", default="slm-synthetic-dpo")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--skip-manifests", action="store_true")
    args = parser.parse_args()
    push_dpo_run(
        dataset_dir=args.dataset_dir,
        repo_owner=args.repo_owner,
        repo_prefix=args.repo_prefix,
        private=args.private,
        env_file=args.env_file,
        run_dir=args.run_dir,
        skip_manifests=args.skip_manifests,
    )


if __name__ == "__main__":
    cli()
