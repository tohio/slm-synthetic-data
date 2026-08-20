"""Push one complete generic DPO run to one Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, create_repo

from slm_synth.accepted_target import discover_run_manifest, require_publish_ready_manifest
from slm_synth.dpo.card import require_dpo_dataset_card_configs
from slm_synth.dpo.report import build_coverage_report, require_publish_ready_report
from slm_synth.dpo.schema import validate_dpo_row
from slm_synth.hf_push import (
    add_file_operation,
    create_dataset_commit,
    legacy_metadata_delete_operations,
)


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Missing HF token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    return token


def discover_jsonl_files(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"DPO dataset directory does not exist: {root}")
    candidates = sorted(path for path in root.rglob("*.jsonl") if path.is_file())
    invalid = [
        path
        for path in candidates
        if path.parent != root or ".batch" in path.stem
    ]
    if invalid:
        rendered = ", ".join(str(path.relative_to(root)) for path in invalid)
        raise ValueError(
            "DPO publication accepts only flat final data/<preference_dimension>.jsonl files; "
            f"remove batch, nested, or compatibility artifacts: {rendered}"
        )
    files = candidates
    if not files:
        raise FileNotFoundError(f"No DPO JSONL files found in {root}")
    return files
def count_and_validate_jsonl(
    path: str | Path, *, expected_preference_dimension: str | None = None
) -> int:
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
            validated = validate_dpo_row(row)
            if (
                expected_preference_dimension is not None
                and validated["metadata"]["preference_dimension"]
                != expected_preference_dimension
            ):
                raise ValueError(
                    f"DPO row {validated['id']} in {jsonl_path} has preference_dimension "
                    f"{validated['metadata']['preference_dimension']!r}; "
                    f"expected {expected_preference_dimension!r}"
                )
            count += 1
    return count


def preference_dimension_from_dataset_path(path: str | Path) -> str:
    return Path(path).stem


def _artifact_manifest_paths(*, run_dir: Path, skip_manifests: bool) -> list[Path]:
    if skip_manifests:
        return []
    manifest_dir = run_dir / "manifests"
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"DPO manifest directory does not exist: {manifest_dir}")
    paths = sorted(manifest_dir.glob("*.manifest.json"))
    if not paths:
        raise FileNotFoundError(f"DPO manifest directory contains no manifests: {manifest_dir}")
    return paths


def _stale_repository_delete_operations(
    api: HfApi,
    *,
    repo_id: str,
    current_data_paths: set[str],
    current_manifest_paths: set[str],
) -> list[CommitOperationDelete]:
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return []
    return [
        CommitOperationDelete(path_in_repo=path)
        for path in repo_files
        if (
            path.startswith("data/")
            and path.endswith(".jsonl")
            and path not in current_data_paths
        )
        or (
            path.startswith("artifacts/manifests/")
            and path.endswith(".manifest.json")
            and path not in current_manifest_paths
        )
    ]


def push_dpo_run(
    *,
    dataset_dir: str | Path,
    repo_id: str,
    private: bool = False,
    env_file: str | None = None,
    run_dir: str | Path | None = None,
    skip_manifests: bool = False,
) -> dict[str, Any]:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    dataset_root = Path(dataset_dir)
    clean_repo_id = repo_id.strip().strip("/") if isinstance(repo_id, str) else ""
    if "/" not in clean_repo_id:
        raise ValueError("repo_id must use the form owner/name")
    if run_dir is None:
        raise ValueError("run_dir is required for DPO acceptance and publish-readiness checks")
    root = Path(run_dir)
    run_manifest = discover_run_manifest(root, dataset_type="dpo")
    require_publish_ready_manifest(run_manifest, artifact_name="DPO")
    manifest_value = json.loads(run_manifest.read_text(encoding="utf-8"))
    metadata = manifest_value.get("metadata", {}) if isinstance(manifest_value, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError("DPO run manifest is missing candidate accounting")
    files = discover_jsonl_files(dataset_root)
    manifest_dimensions = (
        manifest_value.get("preference_dimensions")
        if isinstance(manifest_value, dict)
        else None
    )
    if not isinstance(manifest_dimensions, list) or not all(
        isinstance(dimension, str) and dimension for dimension in manifest_dimensions
    ):
        raise ValueError("DPO run manifest is missing preference_dimensions")
    files_by_dimension: dict[str, list[Path]] = {}
    for file_path in files:
        dimension = preference_dimension_from_dataset_path(file_path)
        files_by_dimension.setdefault(dimension, []).append(file_path)
    expected_dimensions = set(manifest_dimensions)
    if set(files_by_dimension) != expected_dimensions:
        raise ValueError(
            "DPO dataset files do not match run-manifest preference dimensions: "
            f"expected {sorted(expected_dimensions)}, got {sorted(files_by_dimension)}"
        )
    if any(
        len(dimension_files) != 1
        for dimension_files in files_by_dimension.values()
    ):
        raise ValueError(
            "consolidated DPO publishing requires exactly one final JSONL file "
            "per preference dimension"
        )

    coverage_path = root / "coverage.json"
    if not coverage_path.is_file():
        raise FileNotFoundError(f"DPO acceptance report does not exist: {coverage_path}")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not isinstance(coverage, dict):
        raise ValueError(f"DPO acceptance report must contain a JSON object: {coverage_path}")
    require_publish_ready_report(coverage)
    live_report = build_coverage_report(files, require_holdout_check=False)
    require_publish_ready_report(live_report, artifact_name="DPO dataset files")
    if (
        coverage.get("row_count") != live_report["row_count"]
        or coverage.get("content_quality") != live_report["content_quality"]
    ):
        raise ValueError("DPO acceptance report is stale for the current dataset files; rebuild dpo-report")
    acceptance = coverage["acceptance"]
    expected_counts = {
        "attempted_pairs": metadata.get("attempted_pairs"),
        "accepted_pairs": metadata.get("accepted_pairs"),
        "rejected_pairs": metadata.get("rejected_pairs", 0),
        "duplicate_pairs": metadata.get("duplicate_pairs", 0),
        "estimated_tokens": metadata.get("estimated_tokens"),
    }
    if any(acceptance.get(field) != value for field, value in expected_counts.items()):
        raise ValueError("DPO acceptance report does not match run-manifest accounting; rebuild dpo-report")
    if acceptance["accepted_pairs"] != live_report["row_count"]:
        raise ValueError("DPO run manifest accepted count does not match current dataset files")

    readme_path = root / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(f"required DPO dataset card is missing: {readme_path}")
    require_dpo_dataset_card_configs(
        readme_path, preference_dimensions=manifest_dimensions
    )

    token = get_hf_token()
    api = HfApi(token=token)
    create_repo(repo_id=clean_repo_id, repo_type="dataset", private=private, exist_ok=True)

    total_pairs = 0
    uploaded_files: list[str] = []
    data_operations: list[CommitOperationAdd] = []
    for dimension in sorted(files_by_dimension):
        file_path = files_by_dimension[dimension][0]
        pair_count = count_and_validate_jsonl(
            file_path, expected_preference_dimension=dimension
        )
        total_pairs += pair_count
        path_in_repo = f"data/{dimension}.jsonl"
        print(f"[push_hf] staging {file_path} -> {clean_repo_id}/{path_in_repo} pairs={pair_count}")
        data_operations.append(
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(file_path))
        )
        uploaded_files.append(path_in_repo)

    manifest_paths = _artifact_manifest_paths(run_dir=root, skip_manifests=skip_manifests)
    current_data_paths = set(uploaded_files)
    current_manifest_paths = {
        f"artifacts/manifests/{manifest_path.name}" for manifest_path in manifest_paths
    }
    operations = legacy_metadata_delete_operations(api, repo_id=clean_repo_id)
    operations.extend(
        _stale_repository_delete_operations(
            api,
            repo_id=clean_repo_id,
            current_data_paths=current_data_paths,
            current_manifest_paths=current_manifest_paths,
        )
    )
    operations.extend(data_operations)
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_path.read_bytes())
    )
    coverage_op = add_file_operation(
        coverage_path,
        path_in_repo="artifacts/coverage.json",
        required=True,
    )
    if coverage_op is not None:
        operations.append(coverage_op)
    for manifest_path in manifest_paths:
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"artifacts/manifests/{manifest_path.name}",
                path_or_fileobj=str(manifest_path),
            )
        )

    print(f"[push_hf] committing {len(operations)} file operation(s) to {clean_repo_id}")
    create_dataset_commit(
        api,
        repo_id=clean_repo_id,
        operations=operations,
        commit_message="Update consolidated generic DPO dataset",
    )
    result = {
        "repo_id": clean_repo_id,
        "files": uploaded_files,
        "preference_dimensions": sorted(files_by_dimension),
        "preference_dimension_count": len(files_by_dimension),
        "pairs": total_pairs,
    }
    print(
        f"[push_hf] Completed consolidated DPO push repo={clean_repo_id} "
        f"preference_dimensions={result['preference_dimension_count']} "
        f"files={len(uploaded_files)} pairs={total_pairs}"
    )
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Push one consolidated generic DPO dataset to Hugging Face.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--skip-manifests", action="store_true")
    args = parser.parse_args()
    push_dpo_run(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        private=args.private,
        env_file=args.env_file,
        run_dir=args.run_dir,
        skip_manifests=args.skip_manifests,
    )


if __name__ == "__main__":
    cli()
