"""Push SFT run outputs to a Hugging Face dataset repo."""

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
    files = discover_jsonl_files(dataset_root)
    total_rows = 0
    uploaded_files: list[str] = []

    for file_path in files:
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
                upload_optional_file(
                    api,
                    repo_id=repo_id,
                    path=manifest_path,
                    path_in_repo=f"manifests/{manifest_path.name}",
                )

    result = {"repo_id": repo_id, "files": uploaded_files, "rows": total_rows}
    print(f"[push_hf] Completed SFT push repo={repo_id} files={len(uploaded_files)} rows={total_rows}")
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description="Push SFT run outputs to Hugging Face.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--skip-manifests", action="store_true")
    args = parser.parse_args()
    push_sft_run(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        private=args.private,
        env_file=args.env_file,
        run_dir=args.run_dir,
        skip_manifests=args.skip_manifests,
    )


if __name__ == "__main__":
    cli()
