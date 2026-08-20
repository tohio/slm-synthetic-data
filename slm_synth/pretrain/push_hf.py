"""Publish one consolidated, quality-gated pretraining dataset."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, create_repo

from slm_synth.hf_push import create_dataset_commit
from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.pretrain.dedup import (
    DEFAULT_SHINGLE_SIZE,
    PUBLIC_FILENAME,
    audit_public_records,
)
from slm_synth.pretrain.report_diversity import DEFAULT_NEAR_DUPLICATE_THRESHOLD
from slm_synth.pretrain.curate import verify_completion_report


def require_complete_accepted_token_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"accepted-token completion report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("publish_ready") is not True:
        raise ValueError(
            "pretraining run has not reached its accepted-token target: "
            f"status={report.get('status')!r} deficit={report.get('token_deficit')!r}"
        )
    return report


def load_env_file(env_file: str | None = None) -> None:
    load_dotenv(env_file) if env_file else load_dotenv()


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Missing HF token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    return token


def get_export_config(cfg: Mapping[str, Any]) -> tuple[str, bool]:
    export_cfg = cfg.get("export", {}) or {}
    repo_id = str(
        export_cfg.get("hf_repo")
        or export_cfg.get("repo_id")
        or export_cfg.get("repository")
        or ""
    ).strip().strip("/")
    if "/" not in repo_id:
        raise ValueError("pretraining HF repo must use the form owner/name")
    return repo_id, bool(export_cfg.get("private", False))


def iter_public_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"pretraining public row must be an object: {path}:{line_number}")
            yield row


def push_consolidated_dataset(
    *,
    api: HfApi,
    repo_id: str,
    private: bool,
    dataset_path: Path,
    readme_path: Path,
    threshold: float,
    shingle_size: int,
) -> dict[str, Any]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"consolidated pretraining dataset does not exist: {dataset_path}")
    if not readme_path.is_file():
        raise FileNotFoundError(f"pretraining dataset card does not exist: {readme_path}")

    quality = audit_public_records(
        iter_public_jsonl(dataset_path),
        threshold=threshold,
        shingle_size=shingle_size,
    )
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo="data/pretrain.jsonl", path_or_fileobj=str(dataset_path)),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_path)),
    ]
    create_dataset_commit(
        api,
        repo_id=repo_id,
        operations=operations,
        commit_message="Update consolidated synthetic pretraining dataset",
    )
    print(
        f"[push_hf] Completed consolidated pretraining push repo={repo_id} "
        f"rows={quality['rows']} exact_duplicates=0 near_duplicates=0"
    )
    return {"repo_id": repo_id, **quality}


def main(
    *,
    config: str | Path = "configs/synthetic.yaml",
    repo_id: str | None = None,
    private: bool | None = None,
    env_file: str | None = None,
) -> dict[str, Any]:
    load_env_file(env_file)
    cfg = load_yaml_config(config)
    output_dir = resolve_output_dir(cfg)
    configured_repo, configured_private = get_export_config(cfg)
    target_repo = repo_id.strip().strip("/") if repo_id else configured_repo
    if "/" not in target_repo:
        raise ValueError("pretraining HF repo must use the form owner/name")
    target_private = configured_private if private is None else private
    dedup_cfg = cfg.get("dedup", {}) or {}
    verify_completion_report(output_dir, list(cfg.get("mix", {})))
    api = HfApi(token=get_hf_token())
    return push_consolidated_dataset(
        api=api,
        repo_id=target_repo,
        private=bool(target_private),
        dataset_path=output_dir / "deduped" / PUBLIC_FILENAME,
        readme_path=output_dir / "README.md",
        threshold=float(dedup_cfg.get("near_duplicate_threshold", DEFAULT_NEAR_DUPLICATE_THRESHOLD)),
        shingle_size=int(dedup_cfg.get("shingle_size", DEFAULT_SHINGLE_SIZE)),
    )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Push one consolidated, duplicate-free pretraining dataset"
    )
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true", default=None)
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args()
    main(
        config=args.config,
        repo_id=args.repo_id,
        private=args.private,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    cli()
