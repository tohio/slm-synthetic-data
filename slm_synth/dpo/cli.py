"""Command-line helpers for synthetic DPO datasets."""

from __future__ import annotations

import argparse
import json

from slm_synth.throughput_defaults import (
    DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE,
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
)
from slm_synth.run_summary import print_dpo_run_summary
from slm_synth.dpo.generation import generate_llm_batch_from_files, materialize_llm_batch_from_files
from slm_synth.dpo.report import build_coverage_report, write_coverage_report
from slm_synth.dpo.runs import generate_llm_run
from slm_synth.dpo.spec_builders import DPO_SPEC_FAMILIES, build_and_write_specs


def _openrouter_routing_kwargs(args: argparse.Namespace) -> dict[str, str | None]:
    kwargs: dict[str, str | None] = {}
    if getattr(args, "openrouter_routing_mode", None) is not None:
        kwargs["openrouter_routing_mode"] = args.openrouter_routing_mode
    if getattr(args, "openrouter_provider", None) is not None:
        kwargs["openrouter_provider"] = args.openrouter_provider
    return kwargs


def cmd_build_specs(args: argparse.Namespace) -> int:
    count = build_and_write_specs(
        family=args.family,
        count=args.count,
        output_path=args.output,
        start_index=args.start_index,
    )
    print(f"wrote {count} DPO task spec(s) for {args.family} to {args.output}")
    return 0


def cmd_report_coverage(args: argparse.Namespace) -> int:
    report = build_coverage_report(args.input)
    if args.output:
        output_path = write_coverage_report(report=report, path=args.output)
        print(f"wrote DPO coverage report to {output_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_materialize_llm_batch(args: argparse.Namespace) -> int:
    result = materialize_llm_batch_from_files(
        specs_path=args.specs,
        teacher_response_path=args.teacher_response,
        output_path=args.output,
        manifest_path=args.manifest,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        generation_run=args.generation_run,
    )
    print(
        "materialized "
        f"{result.row_count} LLM-generated DPO row(s) to {result.dataset_path}; "
        f"manifest: {result.manifest_path}"
    )
    return 0


def cmd_generate_llm_batch(args: argparse.Namespace) -> int:
    result = generate_llm_batch_from_files(
        specs_path=args.specs,
        output_path=args.output,
        manifest_path=args.manifest,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        generation_run=args.generation_run,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        request_timeout=args.request_timeout,
        max_request_retries=args.max_request_retries,
        max_retryable_request_attempts=args.max_retryable_request_attempts,
        retry_max_elapsed_seconds=args.retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=args.adaptive_maximum_in_flight,
        adaptive_initial_in_flight=args.adaptive_initial_in_flight,
        **_openrouter_routing_kwargs(args),
    )
    print(
        "generated "
        f"{result.row_count} LLM-generated DPO row(s) to {result.dataset_path}; "
        f"manifest: {result.manifest_path}"
    )
    return 0


def cmd_generate_llm_run(args: argparse.Namespace) -> int:
    result = generate_llm_run(
        families=args.families,
        count_per_family=args.count_per_family,
        target_pairs=args.target_pairs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        generation_run=args.generation_run,
        max_tokens=args.max_tokens,
        start_index=args.start_index,
        temperature=args.temperature,
        top_p=args.top_p,
        request_timeout=args.request_timeout,
        max_request_retries=args.max_request_retries,
        max_retryable_request_attempts=args.max_retryable_request_attempts,
        retry_max_elapsed_seconds=args.retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=args.concurrency,
        adaptive_initial_in_flight=args.adaptive_initial_in_flight,
        adaptive_initial_batch_size=args.adaptive_initial_batch_size,
        adaptive_batch_increase_successes=args.adaptive_batch_increase_successes,
        concurrency=args.concurrency,
        max_backfill_rounds=args.max_backfill_rounds,
        run_manifest_filename=args.run_manifest_filename,
        **_openrouter_routing_kwargs(args),
    )
    print(
        "generated "
        f"{result.row_count} LLM-generated DPO row(s) across {len(result.families)} family/families "
        f"for run {result.generation_run}; run manifest: {result.manifest_path}"
    )
    print_dpo_run_summary(result.manifest_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slm_synth.dpo.cli",
        description="Synthetic DPO dataset helpers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_specs_parser = subparsers.add_parser("build-specs")
    build_specs_parser.add_argument("--family", required=True, choices=sorted(DPO_SPEC_FAMILIES))
    build_specs_parser.add_argument("--count", required=True, type=int)
    build_specs_parser.add_argument("--output", required=True, help="Output DPO task spec JSONL path.")
    build_specs_parser.add_argument("--start-index", type=int, default=1)
    build_specs_parser.set_defaults(func=cmd_build_specs)

    llm_batch_parser = subparsers.add_parser("materialize-llm-batch")
    llm_batch_parser.add_argument("--specs", required=True, help="DPO task spec JSONL path.")
    llm_batch_parser.add_argument("--teacher-response", required=True, help="Saved teacher batch response JSON path.")
    llm_batch_parser.add_argument("--output", required=True, help="Output DPO JSONL path.")
    llm_batch_parser.add_argument("--manifest", required=True, help="Output local manifest JSON path.")
    llm_batch_parser.add_argument("--teacher-model", required=True)
    llm_batch_parser.add_argument("--teacher-provider", default="openrouter")
    llm_batch_parser.add_argument("--generation-run", required=True)
    llm_batch_parser.set_defaults(func=cmd_materialize_llm_batch)

    generate_parser = subparsers.add_parser("generate-llm-batch")
    generate_parser.add_argument("--specs", required=True, help="DPO task spec JSONL path.")
    generate_parser.add_argument("--output", required=True, help="Output DPO JSONL path.")
    generate_parser.add_argument("--manifest", required=True, help="Output local manifest JSON path.")
    generate_parser.add_argument("--teacher-model", required=True)
    generate_parser.add_argument("--teacher-provider", default="openrouter")
    generate_parser.add_argument("--generation-run", required=True)
    generate_parser.add_argument("--max-tokens", required=True, type=int)
    generate_parser.add_argument("--temperature", type=float, default=0.2)
    generate_parser.add_argument("--top-p", type=float, default=0.95)
    generate_parser.add_argument("--request-timeout", type=float, default=None)
    generate_parser.add_argument("--max-request-retries", type=int, default=3)
    generate_parser.add_argument("--max-retryable-request-attempts", type=int, default=20)
    generate_parser.add_argument("--retry-max-elapsed-seconds", type=float, default=1800.0)
    generate_parser.add_argument("--adaptive-maximum-in-flight", type=int, default=DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT)
    generate_parser.add_argument("--adaptive-initial-in-flight", type=int, default=DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT)
    generate_parser.add_argument("--openrouter-routing-mode", choices=["auto", "prefer", "strict"], default=None)
    generate_parser.add_argument("--openrouter-provider", default=None)
    generate_parser.set_defaults(func=cmd_generate_llm_batch)

    generate_run_parser = subparsers.add_parser("generate-llm-run")
    generate_run_parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=["all", *sorted(DPO_SPEC_FAMILIES)],
        help="DPO spec families to generate, or 'all'.",
    )
    planning_group = generate_run_parser.add_mutually_exclusive_group(required=True)
    planning_group.add_argument("--count-per-family", type=int)
    planning_group.add_argument("--target-pairs", type=int)
    generate_run_parser.add_argument("--batch-size", required=True, type=int)
    generate_run_parser.add_argument("--output-dir", required=True)
    generate_run_parser.add_argument("--manifest-dir", required=True)
    generate_run_parser.add_argument("--teacher-model", required=True)
    generate_run_parser.add_argument("--teacher-provider", default="openrouter")
    generate_run_parser.add_argument("--generation-run", required=True)
    generate_run_parser.add_argument("--max-tokens", required=True, type=int)
    generate_run_parser.add_argument("--start-index", type=int, default=1)
    generate_run_parser.add_argument("--temperature", type=float, default=0.2)
    generate_run_parser.add_argument("--top-p", type=float, default=0.95)
    generate_run_parser.add_argument("--request-timeout", type=float, default=None)
    generate_run_parser.add_argument("--max-request-retries", type=int, default=3)
    generate_run_parser.add_argument("--max-retryable-request-attempts", type=int, default=20)
    generate_run_parser.add_argument("--retry-max-elapsed-seconds", type=float, default=1800.0)
    generate_run_parser.add_argument("--adaptive-maximum-in-flight", type=int, default=1)
    generate_run_parser.add_argument("--adaptive-initial-in-flight", type=int, default=DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT)
    generate_run_parser.add_argument("--adaptive-initial-batch-size", type=int, default=DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE)
    generate_run_parser.add_argument("--adaptive-batch-increase-successes", type=int, default=DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES)
    generate_run_parser.add_argument("--concurrency", type=int, default=DEFAULT_OPENROUTER_SMOKE_CONCURRENCY)
    generate_run_parser.add_argument("--max-backfill-rounds", type=int, default=2)
    generate_run_parser.add_argument("--run-manifest-filename", default=None)
    generate_run_parser.add_argument("--openrouter-routing-mode", choices=["auto", "prefer", "strict"], default=None)
    generate_run_parser.add_argument("--openrouter-provider", default=None)
    generate_run_parser.set_defaults(func=cmd_generate_llm_run)

    coverage_parser = subparsers.add_parser("report-coverage")
    coverage_parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more DPO JSONL files or directories containing JSONL files.",
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
