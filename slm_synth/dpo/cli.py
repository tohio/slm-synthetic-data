"""Command-line helpers for synthetic DPO datasets."""

from __future__ import annotations

import argparse
import json

from slm_synth.dpo.report import build_coverage_report, write_coverage_report
from slm_synth.taxonomy.holdouts import HoldoutRegistry


def cmd_report_coverage(args: argparse.Namespace) -> int:
    report = build_coverage_report(
        args.input,
        holdout_registry=_load_holdout_registry(args.holdout_registry),
        run_manifest=args.run_manifest,
    )
    if args.output:
        output_path = write_coverage_report(report=report, path=args.output)
        print(f"wrote DPO coverage report to {output_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slm_synth.dpo.cli",
        description="Synthetic DPO dataset reporting helpers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    coverage_parser = subparsers.add_parser("report-coverage")
    coverage_parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more DPO JSONL files or directories containing JSONL files.",
    )
    coverage_parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    coverage_parser.add_argument("--holdout-registry", default=None)
    coverage_parser.add_argument("--run-manifest", default=None)
    coverage_parser.set_defaults(func=cmd_report_coverage)

    return parser


def _load_holdout_registry(path: str | None) -> HoldoutRegistry | None:
    return HoldoutRegistry.from_file(path) if path else None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
