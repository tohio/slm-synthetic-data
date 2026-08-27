"""Command-line reporting helpers for Distillation-SFT."""

from __future__ import annotations

import argparse
import json

from slm_synth.distillation_sft.card import write_dataset_card
from slm_synth.distillation_sft.report import build_coverage_report, write_coverage_report


def cmd_build_dataset_card(args: argparse.Namespace) -> int:
    path = write_dataset_card(
        run_manifest_path=args.run_manifest,
        output_path=args.output,
        dataset_name=args.dataset_name,
        license_name=args.license,
        language=args.language,
    )
    print(f"wrote dataset card to {path}")
    return 0


def cmd_report_coverage(args: argparse.Namespace) -> int:
    report = build_coverage_report(args.run_manifest)
    if args.output:
        output_path = write_coverage_report(report=report, path=args.output)
        print(f"wrote distillation coverage report to {output_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slm_synth.distillation_sft.cli",
        description="Distillation-SFT reporting and dataset-card helpers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    card_parser = subparsers.add_parser("build-dataset-card")
    card_parser.add_argument("--run-manifest", required=True)
    card_parser.add_argument("--output", required=True)
    card_parser.add_argument("--dataset-name", required=True)
    card_parser.add_argument("--license", default=None)
    card_parser.add_argument("--language", default="en")
    card_parser.set_defaults(func=cmd_build_dataset_card)

    coverage_parser = subparsers.add_parser("report-coverage")
    coverage_parser.add_argument("--run-manifest", required=True)
    coverage_parser.add_argument("--output", default=None)
    coverage_parser.set_defaults(func=cmd_report_coverage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
