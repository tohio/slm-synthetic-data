"""Command-line helpers for isolated distillation-DPO artifacts."""

from __future__ import annotations

import argparse
import json

from slm_synth.distillation_dpo.card import write_dataset_card
from slm_synth.distillation_dpo.report import build_coverage_report, write_coverage_report
from slm_synth.distillation_dpo.runs import materialize_production_run, materialize_seed_dataset, materialize_seed_run
from slm_synth.distillation_dpo.seeds import DISTILLATION_DPO_FAMILIES


def cmd_materialize_seed_dataset(args: argparse.Namespace) -> int:
    result = materialize_seed_dataset(
        family=args.family,
        count=args.count,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        generation_run=args.generation_run,
        token_target=args.token_target,
        start_index=args.start_index,
        dataset_filename=args.dataset_filename,
        manifest_filename=args.manifest_filename,
    )
    print(
        "materialized "
        f"{result.row_count} distillation-DPO row(s) for {result.family} to {result.dataset_path}; "
        f"manifest: {result.manifest_path}"
    )
    return 0


def cmd_materialize_seed_run(args: argparse.Namespace) -> int:
    result = materialize_seed_run(
        families=args.families,
        count_per_family=args.count_per_family,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        generation_run=args.generation_run,
        token_target=args.token_target,
        start_index=args.start_index,
        run_manifest_filename=args.run_manifest_filename,
    )
    print(
        "materialized "
        f"{result.row_count} distillation-DPO row(s) across {len(result.families)} family/families "
        f"for run {result.generation_run}; run manifest: {result.manifest_path}"
    )
    return 0


def cmd_materialize_production_run(args: argparse.Namespace) -> int:
    result = materialize_production_run(
        families=args.families,
        target_pairs=args.target_pairs,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        generation_run=args.generation_run,
        start_index=args.start_index,
        run_manifest_filename=args.run_manifest_filename,
    )
    print(
        "materialized "
        f"{result.accepted_pairs} accepted distillation-DPO pair(s) "
        f"from {result.planned_pairs} planned pair(s) across {len(result.families)} family/families "
        f"for run {result.generation_run}; run manifest: {result.manifest_path}"
    )
    return 0


def cmd_report_coverage(args: argparse.Namespace) -> int:
    report = build_coverage_report(args.input)
    if args.output:
        output_path = write_coverage_report(report=report, path=args.output)
        print(f"wrote distillation-DPO coverage report to {output_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_build_dataset_card(args: argparse.Namespace) -> int:
    path = write_dataset_card(
        run_manifest_path=args.run_manifest,
        output_path=args.output,
        dataset_name=args.dataset_name,
        license_name=args.license,
        language=args.language,
    )
    print(f"wrote distillation-DPO dataset card to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slm_synth.distillation_dpo.cli",
        description="Distillation-DPO dataset helpers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    family_choices = sorted(DISTILLATION_DPO_FAMILIES)

    materialize_parser = subparsers.add_parser("materialize-seed-dataset")
    materialize_parser.add_argument("--family", required=True, choices=family_choices)
    materialize_parser.add_argument("--count", required=True, type=int)
    materialize_parser.add_argument("--output-dir", required=True)
    materialize_parser.add_argument("--manifest-dir", required=True)
    materialize_parser.add_argument("--teacher-model", required=True)
    materialize_parser.add_argument("--generation-run", required=True)
    materialize_parser.add_argument("--teacher-provider", default="openrouter")
    materialize_parser.add_argument("--token-target", default=None)
    materialize_parser.add_argument("--start-index", type=int, default=1)
    materialize_parser.add_argument("--dataset-filename", default=None)
    materialize_parser.add_argument("--manifest-filename", default=None)
    materialize_parser.set_defaults(func=cmd_materialize_seed_dataset)

    seed_run_parser = subparsers.add_parser("materialize-seed-run")
    seed_run_parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=["all", *family_choices],
        help="Distillation-DPO families to materialize, or 'all'.",
    )
    seed_run_parser.add_argument("--count-per-family", required=True, type=int)
    seed_run_parser.add_argument("--output-dir", required=True)
    seed_run_parser.add_argument("--manifest-dir", required=True)
    seed_run_parser.add_argument("--teacher-model", required=True)
    seed_run_parser.add_argument("--generation-run", required=True)
    seed_run_parser.add_argument("--teacher-provider", default="openrouter")
    seed_run_parser.add_argument("--token-target", default=None)
    seed_run_parser.add_argument("--start-index", type=int, default=1)
    seed_run_parser.add_argument("--run-manifest-filename", default=None)
    seed_run_parser.set_defaults(func=cmd_materialize_seed_run)

    production_run_parser = subparsers.add_parser("materialize-production-run")
    production_run_parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=["all", *family_choices],
        help="Distillation-DPO families to materialize, or 'all'.",
    )
    production_run_parser.add_argument("--target-pairs", required=True, type=int)
    production_run_parser.add_argument("--output-dir", required=True)
    production_run_parser.add_argument("--manifest-dir", required=True)
    production_run_parser.add_argument("--teacher-model", required=True)
    production_run_parser.add_argument("--generation-run", required=True)
    production_run_parser.add_argument("--teacher-provider", default="openrouter")
    production_run_parser.add_argument("--start-index", type=int, default=1)
    production_run_parser.add_argument("--run-manifest-filename", default=None)
    production_run_parser.set_defaults(func=cmd_materialize_production_run)

    coverage_parser = subparsers.add_parser("report-coverage")
    coverage_parser.add_argument("--input", nargs="+", required=True)
    coverage_parser.add_argument("--output", default=None)
    coverage_parser.set_defaults(func=cmd_report_coverage)

    card_parser = subparsers.add_parser("build-dataset-card")
    card_parser.add_argument("--run-manifest", required=True)
    card_parser.add_argument("--output", required=True)
    card_parser.add_argument("--dataset-name", required=True)
    card_parser.add_argument("--license", default=None)
    card_parser.add_argument("--language", default="en")
    card_parser.set_defaults(func=cmd_build_dataset_card)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
