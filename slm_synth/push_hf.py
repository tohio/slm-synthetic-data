from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

from slm_synth.paths import load_yaml_config, resolve_output_dir


SIGNAL_REPO_SUFFIXES = {
    "arithmetic": "arithmetic",
    "task_code": "task-code",
    "educational_qa_mcq_math": "educational-qa-mcq-math",
    "educational_qa_mcq_general": "educational-qa-mcq-general",
    "factual_restraint": "factual-restraint",
}

SIGNAL_DESCRIPTIONS = {
    "arithmetic": "Integer arithmetic, word problems, comparisons, missing-value problems, and compact reasoning steps.",
    "task_code": "Beginner/intermediate Python tasks with short plans and code snippets.",
    "educational_qa_mcq_math": "Machine-verified mathematical multiple-choice questions with explanations.",
    "educational_qa_mcq_general": "Non-math, self-contained educational multiple-choice questions with explanations.",
    "factual_restraint": "Questions that reward cautious answers and discourage unsupported claims.",
}

SIGNAL_SCHEMAS = {
    "arithmetic": "`type`, `question`, `steps`, `answer`",
    "task_code": "`type`, `task`, `plan`, `code`",
    "educational_qa_mcq_math": "`type`, `question`, `choices`, `correct_index`, `explanation`",
    "educational_qa_mcq_general": "`type`, `question`, `choices`, `correct_index`, `explanation`",
    "factual_restraint": "`type`, `question`, `safe_answer`",
}


def load_env_file(env_file: str | None = None) -> None:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "Missing HF token. Add HF_TOKEN or HUGGINGFACE_HUB_TOKEN to .env, "
            "or export it in the shell."
        )
    return token


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def signal_to_repo_id(base_repo: str, signal: str) -> str:
    if signal not in SIGNAL_REPO_SUFFIXES:
        raise ValueError(
            f"Unknown signal '{signal}'. Expected one of: {', '.join(sorted(SIGNAL_REPO_SUFFIXES))}"
        )
    if "/" not in base_repo:
        raise ValueError(
            f"HF repo must include namespace, e.g. 'tohio/slm-synthetic'. Got: {base_repo!r}"
        )

    namespace, name = base_repo.split("/", 1)
    suffix = SIGNAL_REPO_SUFFIXES[signal]
    if name.endswith(f"-{suffix}"):
        return base_repo
    return f"{namespace}/{name}-{suffix}"


def get_export_config(cfg: dict[str, Any]) -> tuple[str, bool, str]:
    export_cfg = cfg.get("export", {}) or {}
    repo_id = (
        export_cfg.get("hf_repo")
        or export_cfg.get("repo_id")
        or export_cfg.get("repository")
        or "user/repo"
    )
    private = bool(export_cfg.get("private", False))
    license_name = str(export_cfg.get("license", "mit") or "mit").lower()
    return repo_id, private, license_name


def dataset_card_yaml(signal: str, license_name: str) -> str:
    tags = ["synthetic", "llm", "pretraining", "reasoning"]
    if signal == "task_code":
        tags.append("code")
    if signal in {"educational_qa_mcq_math", "educational_qa_mcq_general"}:
        tags.append("educational")
    if signal == "educational_qa_mcq_math":
        tags.append("mathematics")

    metadata = {
        "license": license_name,
        "language": ["en"],
        "pretty_name": f"SLM Synthetic {SIGNAL_REPO_SUFFIXES[signal].replace('-', ' ').title()}",
        "tags": tags,
    }
    return yaml.safe_dump(metadata, sort_keys=False).strip()


def write_dataset_card(
    *,
    output_dir: Path,
    signal: str,
    license_name: str,
    file_path: Path,
) -> Path:
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    generated_date = datetime.now(timezone.utc).date().isoformat()
    record_count = count_jsonl(file_path)
    storage_size = human_size(file_path.stat().st_size if file_path.exists() else 0)

    card_path = manifests_dir / f"README_{signal}.md"
    yaml_header = dataset_card_yaml(signal, license_name)

    title = SIGNAL_REPO_SUFFIXES[signal].replace("-", " ").title()
    description = SIGNAL_DESCRIPTIONS.get(signal, "Synthetic data signal.")
    schema = SIGNAL_SCHEMAS.get(signal, "JSONL records")

    content = f"""---
{yaml_header}
---

# SLM Synthetic {title}

{description}

Generated date: {generated_date}  
Total records: {record_count:,}  
Storage size: {storage_size}

## Files

| File | Records | Size |
|---|---:|---:|
| `{file_path.name}` | {record_count:,} | {storage_size} |

## Record Format

Each row is a JSON object stored as one line in JSONL format.

Schema fields:

{schema}

## Deduplication

The uploaded split is exact-deduplicated. Fuzzy MinHash deduplication is not used for synthetic signals because synthetic examples often share useful structure by design.

## Intended Use

This dataset is intended for SLM data experiments, pretraining/continued-pretraining mixtures, pretraining, and behavior evaluation.

## Limitations

The data is synthetic and should be inspected before use in training or evaluation. It may contain simple, repetitive, or imperfect examples. It should not be treated as a source of authoritative factual knowledge.

## License

This dataset is released under the MIT License.
"""
    card_path.write_text(content, encoding="utf-8")
    return card_path


def discover_signal_files(dedup_dir: Path, signal: str | None = None) -> list[tuple[str, Path]]:
    if signal:
        path = dedup_dir / f"{signal}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing deduped file for signal '{signal}': {path}")
        return [(signal, path)]

    files: list[tuple[str, Path]] = []
    for signal_name in SIGNAL_REPO_SUFFIXES:
        path = dedup_dir / f"{signal_name}.jsonl"
        if path.exists():
            files.append((signal_name, path))

    if not files:
        raise FileNotFoundError(f"No deduped JSONL files found in {dedup_dir}")

    return files


def push_signal_repo(
    *,
    api: HfApi,
    base_repo_id: str,
    private: bool,
    license_name: str,
    output_dir: Path,
    signal: str,
    file_path: Path,
    skip_card: bool,
) -> str:
    repo_id = signal_to_repo_id(base_repo_id, signal)

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    if not skip_card:
        card_path = write_dataset_card(
            output_dir=output_dir,
            signal=signal,
            license_name=license_name,
            file_path=file_path,
        )
        print(f"[push_hf] uploading dataset card {card_path} -> {repo_id}/README.md")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"[push_hf] uploading {file_path} -> {repo_id}/train.jsonl")
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    return repo_id


def main(
    dedup_dir: str | Path | None = None,
    *,
    config: str | Path = "configs/synthetic.yaml",
    repo_id: str | None = None,
    private: bool | None = None,
    signal: str | None = None,
    env_file: str | None = None,
    skip_card: bool = False,
) -> None:
    load_env_file(env_file)
    token = get_hf_token()
    api = HfApi(token=token)

    cfg = load_yaml_config(config)
    output_dir = resolve_output_dir(cfg)
    dedup_path = Path(dedup_dir) if dedup_dir else output_dir / "deduped"

    base_repo_id, cfg_private, license_name = get_export_config(cfg)
    if repo_id:
        base_repo_id = repo_id
    if private is None:
        private = cfg_private

    files = discover_signal_files(dedup_path, signal=signal)

    pushed_repos = []
    for signal_name, path in files:
        pushed_repos.append(
            push_signal_repo(
                api=api,
                base_repo_id=base_repo_id,
                private=bool(private),
                license_name=license_name,
                output_dir=output_dir,
                signal=signal_name,
                file_path=path,
                skip_card=skip_card,
            )
        )

    print(
        "[push_hf] Completed "
        f"repos={len(pushed_repos)}, files={len(files)}, signal={signal or 'all'}"
    )
    for pushed_repo in pushed_repos:
        print(f"[push_hf] repo={pushed_repo}")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Push exact-deduped synthetic data to Hugging Face dataset repos."
    )
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--dedup-dir", default=None)
    parser.add_argument("--repo-id", default=None, help="Base HF repo, e.g. tohio/slm-synthetic")
    parser.add_argument("--private", action="store_true", default=None)
    parser.add_argument("--signal", choices=sorted(SIGNAL_REPO_SUFFIXES), default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--skip-card", action="store_true")
    args = parser.parse_args()

    main(
        dedup_dir=args.dedup_dir,
        config=args.config,
        repo_id=args.repo_id,
        private=args.private,
        signal=args.signal,
        env_file=args.env_file,
        skip_card=args.skip_card,
    )


if __name__ == "__main__":
    cli()
