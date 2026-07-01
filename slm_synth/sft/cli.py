"""Command-line helpers for synthetic SFT datasets."""

from __future__ import annotations

import argparse

from slm_synth.sft.runs import materialize_seed_dataset
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
