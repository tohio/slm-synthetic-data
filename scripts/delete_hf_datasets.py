#!/usr/bin/env python3
# Delete Hugging Face dataset repositories.
#
# Dry-run by default. Pass --yes to perform deletion.
# Deletes dataset repositories only; it does not delete local files.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
# BEGIN .env loading for HF deletion
def _strip_dotenv_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv_if_needed(dotenv_path: str | Path = ".env") -> bool:
    # Load Hugging Face token variables from .env without overriding the shell.
    path = Path(dotenv_path)
    if not path.exists():
        return False

    loaded = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
            continue

        value = _strip_dotenv_value(value)
        if value and not os.environ.get(key):
            os.environ[key] = value
            loaded = True

    return loaded


def _get_hf_token() -> str | None:
    _load_dotenv_if_needed()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
# END .env loading for HF deletion


SFT_DPO_FAMILIES = [
    "ai-concept-explanation",
    "basic-arithmetic-qa",
    "capital-city-qa",
    "clear-sky-color-qa",
    "code-explanation-no-code",
    "code-expression-result",
    "code-generation-function",
    "direct-division",
    "direct-subtraction",
    "function-completion-body-only",
    "list-exact-n-items",
    "private-or-unverifiable-company-fact",
    "repeat-exact-n-times",
    "short-factual-stop-behavior",
]


def planned_repos(args: argparse.Namespace) -> list[str]:
    repos: list[str] = []

    repos.extend(args.repo or [])
    repos.extend(_read_repo_file(args.repo_file))

    if args.include_sft:
        repos.extend(f"{args.namespace}/{args.sft_prefix}-{family}" for family in SFT_DPO_FAMILIES)

    if args.include_dpo:
        repos.extend(f"{args.namespace}/{args.dpo_prefix}-{family}" for family in SFT_DPO_FAMILIES)

    if args.include_distillation:
        repos.append(f"{args.namespace}/slm-synthetic-distillation-sft")
        repos.append(f"{args.namespace}/slm-synthetic-distillation-dpo")

    if args.include_legacy_distillation_dpo:
        repos.append(f"{args.namespace}/slm-synthetic-distillation-dpo-teacher-response-preference")

    return _dedupe_and_validate_repos(repos)


def _read_repo_file(path: str | None) -> list[str]:
    if not path:
        return []

    repos: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        repos.append(value)
    return repos


def _dedupe_and_validate_repos(repos: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()

    for repo in repos:
        repo = repo.strip()
        if not repo or repo in seen:
            continue
        if "/" not in repo:
            raise ValueError(f"invalid repo id {repo!r}; expected namespace/name")
        seen.add(repo)
        unique.append(repo)

    return unique


def delete_repo(api: Any, repo_id: str) -> tuple[bool, str]:
    try:
        if hasattr(api, "repo_exists") and not api.repo_exists(repo_id=repo_id, repo_type="dataset"):
            return False, "missing"

        api.delete_repo(repo_id=repo_id, repo_type="dataset")
        return True, "deleted"
    except Exception as exc:
        text = str(exc)
        if "404" in text or "Not Found" in text or "Repository Not Found" in text:
            return False, "missing"
        return False, f"error: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Delete Hugging Face dataset repositories. Defaults to dry-run. "
            "Pass --yes to perform deletion."
        )
    )
    parser.add_argument("--namespace", default="tohio")
    parser.add_argument("--repo", action="append", help="Exact dataset repo id to delete. Can be repeated.")
    parser.add_argument("--repo-file", help="File containing one dataset repo id per line.")
    parser.add_argument("--include-sft", action="store_true", help="Include slm-synthetic-sft-* family repos.")
    parser.add_argument("--include-dpo", action="store_true", help="Include slm-synthetic-dpo-* family repos.")
    parser.add_argument(
        "--include-distillation",
        action="store_true",
        help="Include slm-synthetic-distillation-sft and slm-synthetic-distillation-dpo.",
    )
    parser.add_argument(
        "--include-legacy-distillation-dpo",
        action="store_true",
        help="Include the old long distillation-DPO repo name.",
    )
    parser.add_argument("--sft-prefix", default="slm-synthetic-sft")
    parser.add_argument("--dpo-prefix", default="slm-synthetic-dpo")
    parser.add_argument("--yes", action="store_true", help="Actually delete. Without this, only prints planned deletions.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        repos = planned_repos(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not repos:
        print("No dataset repos selected.")
        print("Use --repo, --repo-file, --include-sft, --include-dpo, or --include-distillation.")
        return 2

    print("Selected Hugging Face dataset repos:")
    for repo in repos:
        print(f"  - {repo}")

    if not args.yes:
        print()
        print("DRY RUN ONLY. Re-run with --yes to delete these dataset repos.")
        return 0

    token = _get_hf_token()
    if not token:
        print("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required for deletion.", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed in this environment.", file=sys.stderr)
        return 2

    api = HfApi(token=token)

    print()
    print("Deleting dataset repos:")
    failures = 0
    for repo in repos:
        ok, status = delete_repo(api, repo)
        print(f"  - {repo}: {status}")
        if not ok and status != "missing":
            failures += 1

    if failures:
        print()
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print()
    print("Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
