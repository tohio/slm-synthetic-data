"""Push distillation-DPO run outputs to per-family Hugging Face dataset repos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

from slm_synth.distillation_dpo.schema import validate_distillation_dpo_row

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
        raise FileNotFoundError(f"distillation-DPO dataset directory does not exist: {root}")
    candidates = sorted(
        path
        for path in root.rglob("*.jsonl")
        if path.is_file() and not _is_internal_dataset_path(path.relative_to(root))
    )
    files = _prefer_final_public_files(candidates)
    if not files:
        raise FileNotFoundError(f"No distillation-DPO JSONL files found in {root}")
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
            validate_distillation_dpo_row(row)
            count += 1
    return count


def family_from_dataset_path(path: str | Path) -> str:
    stem = Path(path).stem
    return stem.split(".batch", 1)[0]


def slugify_family(family: str) -> str:
    slug = family.strip().lower().replace("_", "-")
    if not slug:
        raise ValueError("family must be a non-empty string")
    return slug


def repo_id_for_family(*, repo_owner: str, repo_prefix: str, family: str) -> str:
    clean_owner = repo_owner.strip()
    clean_prefix = repo_prefix.strip().strip("-/")
    if not clean_owner:
        raise ValueError("repo_owner must be a non-empty string")
    if not clean_prefix:
        raise ValueError("repo_prefix must be a non-empty string")
    return f"{clean_owner}/{clean_prefix}-{slugify_family(family)}"


def upload_optional_file(api: HfApi, *, repo_id: str, path: Path, path_in_repo: str) -> None:
    if not path.exists():
        return
    print(f"[push_hf] uploading {path} -> {repo_id}/{path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )


def push_distillation_dpo_run(
    *,
    dataset_dir: str | Path,
    repo_owner: str,
    repo_prefix: str = "distillation-dpo",
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
    files = discover_jsonl_files(dataset_root)

    repos: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for file_path in files:
        family = family_from_dataset_path(file_path)
        repo_id = repo_id_for_family(repo_owner=repo_owner, repo_prefix=repo_prefix, family=family)
        if family not in repos:
            create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
            repos[family] = {"repo_id": repo_id, "files": [], "rows": 0}

        row_count = count_and_validate_jsonl(file_path)
        total_rows += row_count
        repos[family]["rows"] += row_count
        path_in_repo = f"data/{file_path.relative_to(dataset_root).as_posix()}"
        print(f"[push_hf] uploading {file_path} -> {repo_id}/{path_in_repo} rows={row_count}")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        repos[family]["files"].append(path_in_repo)

    if run_dir is not None:
        root = Path(run_dir)
        manifest_paths = sorted((root / "manifests").glob("*.json")) if (root / "manifests").exists() else []
        for family, repo in repos.items():
            repo_id = repo["repo_id"]
            upload_optional_file(api, repo_id=repo_id, path=root / "README.md", path_in_repo="README.md")
            upload_optional_file(api, repo_id=repo_id, path=root / "coverage.json", path_in_repo="coverage.json")
            if not skip_manifests:
                for manifest_path in manifest_paths:
                    if manifest_path.name.startswith(f"{family}.") or manifest_path.name.endswith(".manifest.json"):
                        upload_optional_file(
                            api,
                            repo_id=repo_id,
                            path=manifest_path,
                            path_in_repo=f"manifests/{manifest_path.name}",
                        )

    result = {"repo_count": len(repos), "rows": total_rows, "repos": repos}
    print(
        "[push_hf] Completed distillation-DPO push "
        f"repos={result['repo_count']} rows={result['rows']}"
    )
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Push distillation-DPO run outputs to Hugging Face.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-owner", required=True)
    parser.add_argument("--repo-prefix", default="distillation-dpo")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--skip-manifests", action="store_true")
    args = parser.parse_args()
    push_distillation_dpo_run(
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
