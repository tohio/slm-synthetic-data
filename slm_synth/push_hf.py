import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from huggingface_hub import HfApi, create_repo, upload_file

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

try:
    from slm_synth.paths import load_yaml_config, resolve_output_dir
except Exception:  # pragma: no cover - keeps this script usable during early bootstrap
    def load_yaml_config(path: str | Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def resolve_output_dir(cfg: Dict[str, Any]) -> Path:
        raw = str(cfg.get("output_dir", "data/runs/default"))
        raw = raw.replace("${DATA_DIR}", os.getenv("DATA_DIR", "data/runs"))
        return Path(os.path.expandvars(os.path.expanduser(raw)))


def load_env_file(env_file: str | None = None) -> None:
    """Load repo .env so HF_TOKEN works without shell export."""
    path = Path(env_file or ".env")
    if load_dotenv is not None:
        load_dotenv(path if path.exists() else None)
        return

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "Missing HF token. Add HF_TOKEN or HUGGINGFACE_HUB_TOKEN to .env, "
            "or export it in the shell."
        )
    return token


def get_hf_config(cfg: Dict[str, Any]) -> Tuple[str, bool, str]:
    hf = cfg.get("hf") or cfg.get("huggingface") or cfg.get("hub") or {}

    repo_id = (
        hf.get("repo_id")
        or cfg.get("repo_id")
        or os.getenv("HF_REPO_ID")
    )
    if not repo_id:
        username = hf.get("username") or os.getenv("HF_USERNAME")
        repo = hf.get("repo") or hf.get("repo_name") or os.getenv("HF_REPO")
        if username and repo:
            repo_id = f"{username}/{repo}"

    if not repo_id:
        raise ValueError(
            "Missing Hugging Face repo id. Set hf.repo_id in configs/synthetic.yaml "
            "or set HF_REPO_ID / HF_USERNAME+HF_REPO."
        )

    private = bool(hf.get("private", True))
    license_name = str(hf.get("license") or cfg.get("license") or "other")
    return repo_id, private, license_name


def jsonl_count(path: Path) -> Tuple[int, int]:
    total = 0
    bad_json = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
    return total, bad_json


def file_stats(files: Iterable[Path]) -> List[Dict[str, Any]]:
    stats = []
    for path in sorted(files):
        total, bad_json = jsonl_count(path)
        stats.append(
            {
                "name": path.name,
                "records": total,
                "bad_json": bad_json,
                "size_bytes": path.stat().st_size,
            }
        )
    return stats


def size_category(total_records: int) -> str:
    if total_records < 1_000:
        return "n<1K"
    if total_records < 10_000:
        return "1K<n<10K"
    if total_records < 100_000:
        return "10K<n<100K"
    if total_records < 1_000_000:
        return "100K<n<1M"
    return "1M<n<10M"


def signal_schema(signal: str) -> str:
    schemas = {
        "arithmetic": "`type`, `question`, `steps`, `answer`",
        "educational_qa_mcq": "`type`, `question`, `choices`, `correct_index`, `explanation`",
        "factual_restraint": "`type`, `question`, `safe_answer`",
        "task_code": "`type`, `task`, `plan`, `code`",
    }
    return schemas.get(signal, "JSONL records; see file contents for fields")


def config_value(cfg: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_dataset_card(
    cfg: Dict[str, Any],
    output_dir: Path,
    dedup_dir: Path,
    stats: List[Dict[str, Any]],
    repo_id: str,
    license_name: str,
) -> str:
    total = sum(item["records"] for item in stats)
    run_name = cfg.get("run_name") or output_dir.name
    model = config_value(cfg, ["backend", "model"], cfg.get("model", "unknown"))
    service_tier = config_value(cfg, ["backend", "service_tier"], cfg.get("service_tier", "unknown"))
    batch_size = config_value(cfg, ["generation", "batch_size"], cfg.get("batch_size", "unknown"))
    parallel_requests = config_value(cfg, ["backend", "parallel_requests"], cfg.get("parallel_requests", "unknown"))
    target_tokens = cfg.get("target_total_tokens", cfg.get("tokens", "unknown"))
    dedup_cfg = cfg.get("dedup", {}) if isinstance(cfg.get("dedup", {}), dict) else {}
    dedup_mode = dedup_cfg.get("mode", "exact")
    enable_fuzzy = dedup_cfg.get("enable_fuzzy", dedup_cfg.get("fuzzy_enabled", False))
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    file_rows = "\n".join(
        f"| `{item['name']}` | {item['records']} | {item['size_bytes']} | {item['bad_json']} |"
        for item in stats
    )
    schema_rows = "\n".join(
        f"| `{Path(item['name']).stem}` | {signal_schema(Path(item['name']).stem)} |"
        for item in stats
    )

    return f"""---
license: {license_name}
language:
- en
task_categories:
- text-generation
- question-answering
pretty_name: SLM Synthetic Data
size_categories:
- {size_category(total)}
tags:
- synthetic
- llm-generated
- jsonl
- educational
- arithmetic
- code
---

# SLM Synthetic Data

This dataset contains synthetic training records generated for a small language model pipeline. It is intended for experimentation with instruction-following, arithmetic reasoning, simple coding tasks, educational multiple-choice questions, and factual-restraint behavior.

Repository: `{repo_id}`  
Run: `{run_name}`  
Generated at: `{generated_at}`  
Total records in uploaded split: **{total}**

## Files

| File | Records | Size bytes | Bad JSON |
|---|---:|---:|---:|
{file_rows}

## Record Schemas

Each file is JSONL with one JSON object per line.

| Signal | Fields |
|---|---|
{schema_rows}

## Generation Configuration

| Setting | Value |
|---|---|
| Generator model | `{model}` |
| Service tier | `{service_tier}` |
| Target tokens | `{target_tokens}` |
| Batch size | `{batch_size}` |
| Parallel requests | `{parallel_requests}` |
| Output directory | `{output_dir}` |

## Processing Pipeline

The uploaded files are from the `deduped/` stage:

1. `raw/` — batched LLM generations using a JSON-object contract.
2. `validated/` — schema-valid records only.
3. `deduped/` — exact-deduplicated records used for upload.

## Deduplication Policy

Synthetic records intentionally share structure, so fuzzy near-duplicate removal can destroy useful variation. This dataset uses the following dedup policy:

| Setting | Value |
|---|---|
| Dedup mode | `{dedup_mode}` |
| Fuzzy dedup enabled | `{enable_fuzzy}` |

For downstream training, prefer the uploaded exact-deduped JSONL files. Avoid fuzzy MinHash dedup on these synthetic signals unless you are intentionally running a separate experiment.

## Intended Use

Suitable uses include:

- Small language model training experiments.
- Pipeline validation for synthetic data generation.
- Instruction-following and response-format experiments.
- Arithmetic, MCQ, factual-restraint, and simple coding behavior checks.

## Limitations

- Records are synthetic and may contain mistakes or low-quality examples.
- The data should not be treated as a source of authoritative factual knowledge.
- Code examples are simple and should be reviewed before use in production settings.
- Synthetic distributions may differ from real user queries.

## Reproducibility

The run was produced with `configs/synthetic.yaml` from this repository. To reproduce a similar run, configure the pipeline, generate raw records, validate, exact-deduplicate, and push to Hugging Face.

```bash
make configure PROFILE=balanced TOKENS=<tokens> BATCH=4 CONCURRENCY=8 SERVICE_TIER=flex
python bootstrap_dirs.py
make generate
make validate
make dedup
make push
```

## License

License metadata is set to `{license_name}`. Review and update this field before public release if a stricter project license is required.
"""


def write_dataset_card(cfg: Dict[str, Any], output_dir: Path, repo_id: str, license_name: str) -> Path:
    dedup_dir = output_dir / "deduped"
    stats = file_stats(dedup_dir.glob("*.jsonl"))
    if not stats:
        raise FileNotFoundError(f"No JSONL files found in dedup directory: {dedup_dir}")

    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    card_path = manifests / "README.md"
    card_path.write_text(
        build_dataset_card(cfg, output_dir, dedup_dir, stats, repo_id, license_name),
        encoding="utf-8",
    )
    return card_path


def main(
    dedup_dir: str | Path,
    *,
    repo_id: str,
    private: bool = True,
    token: str | None = None,
    cfg: Dict[str, Any] | None = None,
    output_dir: Path | None = None,
    license_name: str = "other",
    env_file: str | None = None,
    skip_card: bool = False,
) -> None:
    load_env_file(env_file)
    token = token or get_hf_token()
    dedup_dir = Path(dedup_dir)
    output_dir = output_dir or dedup_dir.parent
    cfg = cfg or {}

    api = HfApi()
    create_repo(repo_id, token=token, exist_ok=True, private=private, repo_type="dataset")

    if not skip_card:
        card_path = write_dataset_card(cfg, output_dir, repo_id, license_name)
        print(f"[push_hf] uploading dataset card {card_path} -> {repo_id}/README.md")
        upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    files = sorted(dedup_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in dedup directory: {dedup_dir}")

    for file in files:
        print(f"[push_hf] uploading {file} -> {repo_id}/{file.name}")
        upload_file(
            path_or_fileobj=str(file),
            path_in_repo=file.name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    print(f"[push_hf] Completed repo_id={repo_id}, files={len(files)}, dataset_card={not skip_card}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Push deduped synthetic JSONL files to Hugging Face Hub.")
    parser.add_argument("dedup_dir", nargs="?", help="Directory containing deduped JSONL files.")
    parser.add_argument("--config", default=None, help="Path to configs/synthetic.yaml.")
    parser.add_argument("--repo-id", default=None, help="HF dataset repo id, e.g. username/repo.")
    parser.add_argument("--private", action="store_true", help="Create/update repo as private.")
    parser.add_argument("--public", action="store_true", help="Create/update repo as public.")
    parser.add_argument("--env-file", default=".env", help="Path to dotenv file. Defaults to .env.")
    parser.add_argument("--skip-card", action="store_true", help="Do not generate/upload README.md dataset card.")
    args = parser.parse_args()

    load_env_file(args.env_file)

    cfg: Dict[str, Any] = {}
    output_dir: Path | None = None
    repo_id = args.repo_id
    private = True
    license_name = "other"

    if args.config:
        cfg = load_yaml_config(args.config)
        output_dir = resolve_output_dir(cfg)
        cfg_repo_id, cfg_private, license_name = get_hf_config(cfg)
        repo_id = repo_id or cfg_repo_id
        private = cfg_private

    if args.private:
        private = True
    if args.public:
        private = False

    if not repo_id:
        repo_id, private, license_name = get_hf_config(cfg)

    if args.dedup_dir:
        dedup_dir = Path(args.dedup_dir)
        output_dir = output_dir or dedup_dir.parent
    else:
        if output_dir is None:
            raise ValueError("Provide dedup_dir or --config.")
        dedup_dir = output_dir / "deduped"

    main(
        dedup_dir,
        repo_id=repo_id,
        private=private,
        cfg=cfg,
        output_dir=output_dir,
        license_name=license_name,
        env_file=args.env_file,
        skip_card=args.skip_card,
    )


if __name__ == "__main__":
    cli()
