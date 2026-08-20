from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_delete_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "delete_hf_datasets.py"
    spec = importlib.util.spec_from_file_location("delete_hf_datasets", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planned_repos_includes_distillation_short_names() -> None:
    module = _load_delete_module()
    args = argparse.Namespace(
        namespace="tohio",
        repo=None,
        repo_file=None,
        include_sft=False,
        include_dpo=False,
        include_distillation=True,
        include_legacy_distillation_dpo=False,
        sft_prefix="slm-synthetic-sft",
        dpo_prefix="slm-synthetic-dpo",
    )

    assert module.planned_repos(args) == [
        "tohio/slm-synthetic-distillation-sft",
        "tohio/slm-synthetic-distillation-dpo",
    ]


def test_planned_repos_accepts_exact_generic_repositories_without_duplicates() -> None:
    module = _load_delete_module()
    args = argparse.Namespace(
        namespace="tohio",
        repo=["tohio/slm-synthetic-sft", "tohio/slm-synthetic-dpo"],
        repo_file=None,
        include_sft=True,
        include_dpo=True,
        include_distillation=False,
        include_legacy_distillation_dpo=False,
        sft_prefix="slm-synthetic-sft",
        dpo_prefix="slm-synthetic-dpo",
    )

    repos = module.planned_repos(args)

    assert repos == ["tohio/slm-synthetic-sft", "tohio/slm-synthetic-dpo"]


def test_generic_family_repo_delete_selectors_are_removed() -> None:
    module = _load_delete_module()

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--include-sft"])
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--include-dpo"])


def test_dry_run_does_not_require_hf_token(capsys) -> None:
    module = _load_delete_module()

    rc = module.main(["--namespace", "tohio", "--include-distillation"])

    output = capsys.readouterr().out
    assert rc == 0
    assert "DRY RUN ONLY" in output
    assert "tohio/slm-synthetic-distillation-sft" in output


def test_invalid_repo_id_returns_usage_error(capsys) -> None:
    module = _load_delete_module()

    rc = module.main(["--repo", "not-a-full-repo-id"])

    error = capsys.readouterr().err
    assert rc == 2
    assert "expected namespace/name" in error
