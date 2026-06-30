"""Command-line helpers for response-distillation materialization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from slm_synth.distillation.batches import render_teacher_batch_prompt
from slm_synth.distillation.budget import DEFAULT_ESTIMATED_TOKENS_PER_ROW, build_token_budget_plan
from slm_synth.distillation.prompts import validate_prompt_record
from slm_synth.distillation.generation import generate_and_materialize_signal_batch
from slm_synth.distillation.orchestration import generate_seed_multi_signal_run
from slm_synth.distillation.runs import materialize_teacher_batch
from slm_synth.distillation.seeds import build_seed_prompt_records
from slm_synth.distillation.signals import DISTILLATION_SIGNALS, validate_signal


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    input_path = Path(path)
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {input_path} at line {line_number}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"JSONL record in {input_path} at line {line_number} must be an object")
        records.append(validate_prompt_record(value))
    return records


def _write_jsonl_unvalidated(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """Write internal prompt records to JSONL without public-row validation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validated = validate_prompt_record(row)
            handle.write(json.dumps(validated, ensure_ascii=False) + "\n")
            count += 1
    return count


def cmd_build_seed_prompts(args: argparse.Namespace) -> int:
    signal = validate_signal(args.signal)
    records = build_seed_prompt_records(signal=signal, count=args.count, start_index=args.start_index)
    count = _write_jsonl_unvalidated(records, args.output)
    print(f"wrote {count} {signal} prompt record(s) to {args.output}")
    return 0


def cmd_render_teacher_prompt(args: argparse.Namespace) -> int:
    signal = validate_signal(args.signal)
    prompt_records = _read_jsonl(args.prompts)
    rendered = render_teacher_batch_prompt(signal=signal, prompt_records=prompt_records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"wrote teacher batch prompt to {output_path}")
    return 0


def cmd_materialize_batch(args: argparse.Namespace) -> int:
    signal = validate_signal(args.signal)
    prompt_records = _read_jsonl(args.prompts)
    teacher_response = _read_json(args.teacher_response)
    if not isinstance(teacher_response, Mapping):
        raise ValueError("teacher response file must contain a JSON object")

    result = materialize_teacher_batch(
        signal=signal,
        prompt_records=prompt_records,
        teacher_response=teacher_response,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        generation_run=args.generation_run,
        token_target=args.token_target,
        dataset_filename=args.dataset_filename,
        manifest_filename=args.manifest_filename,
        metadata={"source_prompt_file": str(Path(args.prompts))},
    )
    print(
        "materialized "
        f"{result.row_count} {result.signal} row(s) to {result.dataset_path}; "
        f"manifest: {result.manifest_path}"
    )
    return 0


def cmd_generate_batch(args: argparse.Namespace) -> int:
    signal = validate_signal(args.signal)
    prompt_records = _read_jsonl(args.prompts)
    result = generate_and_materialize_signal_batch(
        signal=signal,
        prompt_records=prompt_records,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        teacher_model=args.teacher_model,
        generation_run=args.generation_run,
        max_tokens=args.max_tokens,
        token_target=args.token_target,
        dataset_filename=args.dataset_filename,
        manifest_filename=args.manifest_filename,
        temperature=args.temperature,
        top_p=args.top_p,
        request_timeout=args.request_timeout,
        max_request_retries=args.max_request_retries,
        max_retryable_request_attempts=args.max_retryable_request_attempts,
        retry_max_elapsed_seconds=args.retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=args.adaptive_maximum_in_flight,
        adaptive_initial_in_flight=args.adaptive_initial_in_flight,
    )
    print(
        "generated and materialized "
        f"{result.row_count} {result.signal} row(s) to {result.dataset_path}; "
        f"manifest: {result.manifest_path}"
    )
    return 0


def cmd_plan_token_target(args: argparse.Namespace) -> int:
    signals = args.signals if args.signals else None
    plan = build_token_budget_plan(
        target=args.target,
        signals=signals,
        estimated_tokens_per_row=args.estimated_tokens_per_row,
    )
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_generate_seed_run(args: argparse.Namespace) -> int:
    signals = args.signals if args.signals else None
    counts_by_signal = None
    token_target = args.token_target
    if args.target_preset is not None:
        plan = build_token_budget_plan(
            target=args.target_preset,
            signals=signals,
            estimated_tokens_per_row=args.estimated_tokens_per_row,
        )
        counts_by_signal = plan.counts_by_signal
        if token_target is None:
            token_target = plan.target_label

    result = generate_seed_multi_signal_run(
        signals=signals,
        count_per_signal=args.count_per_signal,
        counts_by_signal=counts_by_signal,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        teacher_model=args.teacher_model,
        generation_run=args.generation_run,
        max_tokens=args.max_tokens,
        token_target=token_target,
        start_index=args.start_index,
        temperature=args.temperature,
        top_p=args.top_p,
        request_timeout=args.request_timeout,
        max_request_retries=args.max_request_retries,
        max_retryable_request_attempts=args.max_retryable_request_attempts,
        retry_max_elapsed_seconds=args.retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=args.adaptive_maximum_in_flight,
        adaptive_initial_in_flight=args.adaptive_initial_in_flight,
        run_manifest_filename=args.run_manifest_filename,
    )
    signals_text = ", ".join(result.signals)
    print(
        "generated and materialized "
        f"{result.row_count} row(s) across {len(result.results)} signal(s): {signals_text}; "
        f"run manifest: {result.manifest_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slm_synth.distillation.cli",
        description="Non-network helpers for response-distillation prompt and dataset materialization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    signal_choices = sorted(DISTILLATION_SIGNALS)

    seed_parser = subparsers.add_parser("build-seed-prompts")
    seed_parser.add_argument("--signal", required=True, choices=signal_choices)
    seed_parser.add_argument("--count", required=True, type=int)
    seed_parser.add_argument("--output", required=True)
    seed_parser.add_argument("--start-index", type=int, default=1)
    seed_parser.set_defaults(func=cmd_build_seed_prompts)

    render_parser = subparsers.add_parser("render-teacher-prompt")
    render_parser.add_argument("--signal", required=True, choices=signal_choices)
    render_parser.add_argument("--prompts", required=True)
    render_parser.add_argument("--output", required=True)
    render_parser.set_defaults(func=cmd_render_teacher_prompt)

    materialize_parser = subparsers.add_parser("materialize-batch")
    materialize_parser.add_argument("--signal", required=True, choices=signal_choices)
    materialize_parser.add_argument("--prompts", required=True)
    materialize_parser.add_argument("--teacher-response", required=True)
    materialize_parser.add_argument("--output-dir", required=True)
    materialize_parser.add_argument("--manifest-dir", required=True)
    materialize_parser.add_argument("--teacher-model", required=True)
    materialize_parser.add_argument("--generation-run", required=True)
    materialize_parser.add_argument("--teacher-provider", default="openrouter")
    materialize_parser.add_argument("--token-target", default=None)
    materialize_parser.add_argument("--dataset-filename", default=None)
    materialize_parser.add_argument("--manifest-filename", default=None)
    materialize_parser.set_defaults(func=cmd_materialize_batch)


    generate_parser = subparsers.add_parser("generate-batch")
    generate_parser.add_argument("--signal", required=True, choices=signal_choices)
    generate_parser.add_argument("--prompts", required=True)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.add_argument("--manifest-dir", required=True)
    generate_parser.add_argument("--teacher-model", required=True)
    generate_parser.add_argument("--generation-run", required=True)
    generate_parser.add_argument("--max-tokens", type=int, required=True)
    generate_parser.add_argument("--token-target", default=None)
    generate_parser.add_argument("--dataset-filename", default=None)
    generate_parser.add_argument("--manifest-filename", default=None)
    generate_parser.add_argument("--temperature", type=float, default=0.2)
    generate_parser.add_argument("--top-p", type=float, default=0.95)
    generate_parser.add_argument("--request-timeout", type=float, default=None)
    generate_parser.add_argument("--max-request-retries", type=int, default=3)
    generate_parser.add_argument("--max-retryable-request-attempts", type=int, default=20)
    generate_parser.add_argument("--retry-max-elapsed-seconds", type=float, default=1800.0)
    generate_parser.add_argument("--adaptive-maximum-in-flight", type=int, default=1)
    generate_parser.add_argument("--adaptive-initial-in-flight", type=int, default=1)
    generate_parser.set_defaults(func=cmd_generate_batch)


    plan_parser = subparsers.add_parser("plan-token-target")
    plan_parser.add_argument("--target", required=True)
    plan_parser.add_argument("--signals", nargs="+", choices=signal_choices, default=None)
    plan_parser.add_argument("--estimated-tokens-per-row", type=int, default=DEFAULT_ESTIMATED_TOKENS_PER_ROW)
    plan_parser.set_defaults(func=cmd_plan_token_target)

    seed_run_parser = subparsers.add_parser("generate-seed-run")
    seed_run_parser.add_argument("--signals", nargs="+", choices=signal_choices, default=None)
    seed_run_parser.add_argument("--count-per-signal", type=int, default=None)
    seed_run_parser.add_argument("--target-preset", default=None)
    seed_run_parser.add_argument("--estimated-tokens-per-row", type=int, default=DEFAULT_ESTIMATED_TOKENS_PER_ROW)
    seed_run_parser.add_argument("--output-dir", required=True)
    seed_run_parser.add_argument("--manifest-dir", required=True)
    seed_run_parser.add_argument("--teacher-model", required=True)
    seed_run_parser.add_argument("--generation-run", required=True)
    seed_run_parser.add_argument("--max-tokens", type=int, required=True)
    seed_run_parser.add_argument("--token-target", default=None)
    seed_run_parser.add_argument("--start-index", type=int, default=1)
    seed_run_parser.add_argument("--temperature", type=float, default=0.2)
    seed_run_parser.add_argument("--top-p", type=float, default=0.95)
    seed_run_parser.add_argument("--request-timeout", type=float, default=None)
    seed_run_parser.add_argument("--max-request-retries", type=int, default=3)
    seed_run_parser.add_argument("--max-retryable-request-attempts", type=int, default=20)
    seed_run_parser.add_argument("--retry-max-elapsed-seconds", type=float, default=1800.0)
    seed_run_parser.add_argument("--adaptive-maximum-in-flight", type=int, default=1)
    seed_run_parser.add_argument("--adaptive-initial-in-flight", type=int, default=1)
    seed_run_parser.add_argument("--run-manifest-filename", default=None)
    seed_run_parser.set_defaults(func=cmd_generate_seed_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
