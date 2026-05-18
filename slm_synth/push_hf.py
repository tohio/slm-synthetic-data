from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import create_repo, upload_file

from slm_synth.paths import deduped_dir_from_config, load_yaml_config, repo_id_from_config_or_env


def main(dedup_dir: str | Path, repo_id: str, private: bool = False) -> None:
    dedup_dir = Path(dedup_dir)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if not token:
        raise ValueError("Missing HF token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    if not repo_id:
        raise ValueError("Missing repo_id.")
    if not dedup_dir.exists():
        raise FileNotFoundError(f"dedup_dir does not exist: {dedup_dir}")

    files = sorted(dedup_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in dedup_dir: {dedup_dir}")

    create_repo(repo_id, token=token, exist_ok=True, private=private, repo_type="dataset")

    for file in files:
        print(f"[push_hf] uploading {file} -> {repo_id}/{file.name}")
        upload_file(
            path_or_fileobj=str(file),
            path_in_repo=file.name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    print(f"[push_hf] Completed repo_id={repo_id}, files={len(files)}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Push deduped synthetic JSONL files to Hugging Face Hub.")
    parser.add_argument("dedup_dir", nargs="?", help="Path to deduped JSONL directory.")
    parser.add_argument("--config", default=None, help="Path to configs/synthetic.yaml.")
    parser.add_argument("--repo-id", default=None, help="HF repo id, e.g. user/dataset-name.")
    parser.add_argument("--private", action="store_true", help="Create repo as private if it does not exist.")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg = load_yaml_config(args.config)
        dedup_dir = deduped_dir_from_config(args.config)
    elif args.dedup_dir:
        dedup_dir = Path(args.dedup_dir)
    else:
        parser.error("provide either dedup_dir or --config")

    hf_cfg = cfg.get("hf") or cfg.get("huggingface") or {}
    repo_id = args.repo_id or repo_id_from_config_or_env(cfg)
    private = bool(args.private or hf_cfg.get("private", False))
    main(dedup_dir, repo_id=repo_id, private=private)


if __name__ == "__main__":
    cli()
