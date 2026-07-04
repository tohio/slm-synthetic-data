"""Push SFT run outputs to per-family Hugging Face dataset repos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

from slm_synth.sft.schema import validate_sft_row


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Missing HF token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    return token


def discover_jsonl_files(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"SFT dataset directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*.jsonl") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No SFT JSONL files found in {root}")
    return files


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
            validate_sft_row(row)
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


def push_sft_run(
    *,
    dataset_dir: str | Path,
    repo_owner: str,
    repo_prefix: str = "slm-sft",
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

        for file_path in family_files:
            row_count = count_and_validate_jsonl(file_path)
            total_rows += row_count
            path_in_repo = f"data/{file_path.relative_to(dataset_root).as_posix()}"
            print(f"[push_hf] uploading {file_path} -> {repo_id}/{path_in_repo} rows={row_count}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            uploaded_files.append(path_in_repo)

        if run_dir is not None:
            root = Path(run_dir)
            upload_optional_file(api, repo_id=repo_id, path=root / "README.md", path_in_repo="README.md")
            upload_optional_file(api, repo_id=repo_id, path=root / "coverage.json", path_in_repo="coverage.json")
            if not skip_manifests:
                manifests = sorted((root / "manifests").glob("*.json")) if (root / "manifests").exists() else []
                for manifest_path in manifests:
                    if manifest_path.name.startswith(f"{family}.") or manifest_path.name == f"{root.name}.manifest.json":
                        upload_optional_file(
                            api,
                            repo_id=repo_id,
                            path=manifest_path,
                            path_in_repo=f"manifests/{manifest_path.name}",
                        )

        repos[family] = {"repo_id": repo_id, "files": uploaded_files, "rows": total_rows}
        print(f"[push_hf] Completed SFT push repo={repo_id} files={len(uploaded_files)} rows={total_rows}")

    result = {"repos": repos, "repo_count": len(repos), "rows": sum(item["rows"] for item in repos.values())}
    print(f"[push_hf] Completed SFT family push repos={len(repos)} rows={result['rows']}")
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Push SFT run outputs to per-family Hugging Face dataset repos.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-owner", required=True)
    parser.add_argument("--repo-prefix", default="slm-sft")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--skip-manifests", action="store_true")
    args = parser.parse_args()
    push_sft_run(
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
