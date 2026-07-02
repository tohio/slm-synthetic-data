"""Command-line helpers for synthetic SFT datasets."""

from __future__ import annotations

import argparse
import json

from slm_synth.sft.report import build_coverage_report, write_coverage_report
from slm_synth.sft.runs import materialize_seed_dataset, materialize_seed_run
from slm_synth.sft.seeds import SFT_SEED_FAMILIES


def cmd_materialize_seed_dataset(args: argparse.Namespace) -> int:
    result = materialize_seed_dataset(
        family=args.family,
        count=args.count,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        generation_run=args.generation_run,
        start_index=args.start_index,
        dataset_filename=args.dataset_filename,
        manifest_filename=args.manifest_filename,
    )
    print(
        "materialized "
        f"{result.row_count} SFT row(s) for {result.family} to {result.dataset_path}; "
        f"manifest: {result.manifest_path}"
    )
    return 0


def cmd_report_coverage(args: argparse.Namespace) -> int:
    report = build_coverage_report(args.input)
    if args.output:
        output_path = write_coverage_report(report=report, path=args.output)
        print(f"wrote SFT coverage report to {output_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_materialize_seed_run(args: argparse.Namespace) -> int:
    result = materialize_seed_run(
        families=args.families,
        count_per_family=args.count_per_family,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        generation_run=args.generation_run,
        start_index=args.start_index,
        run_manifest_filename=args.run_manifest_filename,
    )
    print(
        "materialized "
        f"{result.row_count} SFT row(s) across {len(result.families)} family/families "
        f"for run {result.generation_run}; run manifest: {result.manifest_path}"
    )
    for family_result in result.results:
        print(
            f"- {family_result.family}: {family_result.row_count} row(s), "
            f"dataset: {family_result.dataset_path}, manifest: {family_result.manifest_path}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slm_synth.sft.cli",
        description="Synthetic SFT dataset helpers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize_parser = subparsers.add_parser("materialize-seed-dataset")
    materialize_parser.add_argument("--family", required=True, choices=sorted(SFT_SEED_FAMILIES))
    materialize_parser.add_argument("--count", required=True, type=int)
    materialize_parser.add_argument("--output-dir", required=True)
    materialize_parser.add_argument("--manifest-dir", required=True)
    materialize_parser.add_argument("--generation-run", required=True)
    materialize_parser.add_argument("--start-index", type=int, default=1)
    materialize_parser.add_argument("--dataset-filename", default=None)
    materialize_parser.add_argument("--manifest-filename", default=None)
    materialize_parser.set_defaults(func=cmd_materialize_seed_dataset)

    seed_run_parser = subparsers.add_parser("materialize-seed-run")
    seed_run_parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=["all", *sorted(SFT_SEED_FAMILIES)],
        help="SFT seed families to materialize, or 'all'.",
    )
    seed_run_parser.add_argument("--count-per-family", required=True, type=int)
    seed_run_parser.add_argument("--output-dir", required=True)
    seed_run_parser.add_argument("--manifest-dir", required=True)
    seed_run_parser.add_argument("--generation-run", required=True)
    seed_run_parser.add_argument("--start-index", type=int, default=1)
    seed_run_parser.add_argument("--run-manifest-filename", default=None)
    seed_run_parser.set_defaults(func=cmd_materialize_seed_run)

    coverage_parser = subparsers.add_parser("report-coverage")
    coverage_parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more SFT JSONL files or directories containing JSONL files.",
    )
    coverage_parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    coverage_parser.set_defaults(func=cmd_report_coverage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
