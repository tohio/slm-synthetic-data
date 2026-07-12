"""Push distillation run outputs to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, create_repo

from slm_synth.accepted_target import require_publish_ready_manifest
from slm_synth.distillation_sft.schema import validate_public_row
from slm_synth.distillation_sft.prompt_quality import normalize_prompt_text
from slm_synth.hf_push import (
    add_file_operation,
    create_dataset_commit,
    dataset_card_bytes,
    legacy_metadata_delete_operations,
)


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



DEFAULT_DISTILLATION_SFT_MIN_UNIQUE_PROMPTS = 1500
DEFAULT_DISTILLATION_SFT_MIN_UNIQUE_PROMPT_RATIO = 0.75


def _prompt_uniqueness_thresholds_from_env() -> tuple[int, float]:
    min_unique_prompts = int(
        os.getenv(
            "DISTILLATION_SFT_MIN_UNIQUE_PROMPTS",
            str(DEFAULT_DISTILLATION_SFT_MIN_UNIQUE_PROMPTS),
        )
    )
    min_unique_ratio = float(
        os.getenv(
            "DISTILLATION_SFT_MIN_UNIQUE_RATIO",
            str(DEFAULT_DISTILLATION_SFT_MIN_UNIQUE_PROMPT_RATIO),
        )
    )
    if min_unique_prompts < 0:
        raise ValueError("DISTILLATION_SFT_MIN_UNIQUE_PROMPTS must be non-negative")
    if not 0 <= min_unique_ratio <= 1:
        raise ValueError("DISTILLATION_SFT_MIN_UNIQUE_RATIO must be between 0 and 1")
    return min_unique_prompts, min_unique_ratio


def build_prompt_uniqueness_summary(files: list[Path]) -> dict[str, Any]:
    prompt_counts: dict[str, int] = {}
    row_count = 0

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL in {file_path} at line {line_number}: {exc}") from exc
                validated = validate_public_row(row)
                key = normalize_prompt_text(validated["prompt"])
                prompt_counts[key] = prompt_counts.get(key, 0) + 1
                row_count += 1

    unique_prompt_count = len(prompt_counts)
    duplicate_prompt_count = row_count - unique_prompt_count
    unique_prompt_ratio = unique_prompt_count / row_count if row_count else 0.0
    duplicate_examples = [
        prompt for prompt, count in sorted(prompt_counts.items()) if count > 1
    ][:10]

    return {
        "row_count": row_count,
        "unique_prompt_count": unique_prompt_count,
        "duplicate_prompt_count": duplicate_prompt_count,
        "unique_prompt_ratio": unique_prompt_ratio,
        "duplicate_examples": duplicate_examples,
    }


def require_publish_prompt_uniqueness(
    files: list[Path],
    *,
    min_unique_prompts: int | None = None,
    min_unique_ratio: float | None = None,
) -> dict[str, Any]:
    if min_unique_prompts is None or min_unique_ratio is None:
        env_min_unique_prompts, env_min_unique_ratio = _prompt_uniqueness_thresholds_from_env()
        if min_unique_prompts is None:
            min_unique_prompts = env_min_unique_prompts
        if min_unique_ratio is None:
            min_unique_ratio = env_min_unique_ratio

    if min_unique_prompts < 0:
        raise ValueError("min_unique_prompts must be non-negative")
    if not 0 <= min_unique_ratio <= 1:
        raise ValueError("min_unique_ratio must be between 0 and 1")

    summary = build_prompt_uniqueness_summary(files)
    failures: list[str] = []
    if summary["row_count"] >= min_unique_prompts and summary["unique_prompt_count"] < min_unique_prompts:
        failures.append(
            "unique prompts "
            f"{summary['unique_prompt_count']} below required minimum {min_unique_prompts}"
        )
    if summary["row_count"] and summary["unique_prompt_ratio"] < min_unique_ratio:
        failures.append(
            "unique prompt ratio "
            f"{summary['unique_prompt_ratio']:.3f} below required minimum {min_unique_ratio:.3f}"
        )
    if failures:
        examples = ", ".join(repr(example) for example in summary["duplicate_examples"][:3])
        detail = "; ".join(failures)
        if examples:
            detail = f"{detail}; duplicate prompt examples: {examples}"
        raise ValueError(f"distillation-SFT prompt uniqueness gate failed: {detail}")
    return summary

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Missing HF token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    return token


def discover_jsonl_files(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"distillation dataset directory does not exist: {root}")
    candidates = sorted(
        path
        for path in root.rglob("*.jsonl")
        if path.is_file() and not _is_internal_dataset_path(path.relative_to(root))
    )
    files = _prefer_final_public_files(candidates)
    if not files:
        raise FileNotFoundError(f"No distillation JSONL files found in {root}")
    return files


def _is_internal_dataset_path(relative_path: Path) -> bool:
    return any(part in INTERNAL_DATASET_DIR_NAMES for part in relative_path.parts[:-1])


def _dataset_key(path: Path) -> str:
    return path.stem.split(".batch", 1)[0]


def _prefer_final_public_files(paths: list[Path]) -> list[Path]:
    files_by_key: dict[str, list[Path]] = {}
    for path in paths:
        files_by_key.setdefault(_dataset_key(path), []).append(path)

    files: list[Path] = []
    for key_paths in files_by_key.values():
        final_files = [path for path in key_paths if ".batch" not in path.stem]
        files.extend(final_files or key_paths)
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
            validate_public_row(row)
            count += 1
    return count


def discover_run_manifest(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    manifest_dir = root / "manifests"
    if not manifest_dir.exists():
        raise FileNotFoundError(f"distillation manifest directory does not exist: {manifest_dir}")

    expected = manifest_dir / f"{root.name}.manifest.json"
    if expected.is_file():
        return expected

    run_manifests: list[Path] = []
    for manifest_path in sorted(manifest_dir.glob("*.manifest.json")):
        if _is_run_manifest(manifest_path):
            run_manifests.append(manifest_path)

    if len(run_manifests) == 1:
        return run_manifests[0]
    if not run_manifests:
        raise FileNotFoundError(
            f"No distillation run manifest found in {manifest_dir}; expected {expected.name}"
        )
    names = ", ".join(path.name for path in run_manifests)
    raise ValueError(f"Multiple distillation run manifests found in {manifest_dir}: {names}")


def _is_run_manifest(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid distillation manifest JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"distillation manifest must contain a JSON object: {path}")
    return isinstance(value.get("datasets"), list)


def push_distillation_run(
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

    token = get_hf_token()
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    dataset_root = Path(dataset_dir)
    root = Path(run_dir) if run_dir is not None else None
    run_manifest: Path | None = None
    if root is not None:
        run_manifest = discover_run_manifest(root)
        require_publish_ready_manifest(run_manifest, artifact_name="distillation SFT")
    files = discover_jsonl_files(dataset_root)
    prompt_uniqueness = require_publish_prompt_uniqueness(files)
    print(
        "[push_hf] distillation-SFT prompt uniqueness "
        f"rows={prompt_uniqueness['row_count']} "
        f"unique_prompts={prompt_uniqueness['unique_prompt_count']} "
        f"unique_ratio={prompt_uniqueness['unique_prompt_ratio']:.3f}"
    )
    total_rows = 0
    uploaded_files: list[str] = []
    operations = legacy_metadata_delete_operations(api, repo_id=repo_id)

    for file_path in files:
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
        coverage_op = add_file_operation(root / "coverage.json", path_in_repo="artifacts/coverage.json", required=True)
        if coverage_op is not None:
            operations.append(coverage_op)
        if not skip_manifests:
            if run_manifest is None:
                raise FileNotFoundError("distillation run manifest is required")
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"artifacts/manifests/{run_manifest.name}",
                    path_or_fileobj=str(run_manifest),
                )
            )

    print(f"[push_hf] committing {len(operations)} file operation(s) to {repo_id}")
    create_dataset_commit(
        api,
        repo_id=repo_id,
        operations=operations,
        commit_message="Update distillation SFT dataset",
    )

    result = {"repo_id": repo_id, "files": uploaded_files, "rows": total_rows}
    print(f"[push_hf] Completed distillation push repo={repo_id} files={len(uploaded_files)} rows={total_rows}")
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Push distillation run outputs to Hugging Face.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--skip-manifests", action="store_true")
    args = parser.parse_args()
    push_distillation_run(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        private=args.private,
        env_file=args.env_file,
        run_dir=args.run_dir,
        skip_manifests=args.skip_manifests,
    )


if __name__ == "__main__":
    cli()
