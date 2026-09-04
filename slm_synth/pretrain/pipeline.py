from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.pretrain.dedup import build_public_record
from slm_synth.pretrain.generate import _grounded_token_target
from slm_synth.pretrain.quality import JUDGE_CRITERIA, run_quality
from slm_synth.pretrain.record_quality import signal_to_filename
from slm_synth.runtime import write_json


DEFAULT_CHARS_PER_TOKEN = 4.0


def _run_curate(
    *,
    config_path: Path,
    signal: str | None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "slm_synth.pretrain.curate",
        "--config",
        str(config_path),
        "--allow-shortfall",
    ]
    if signal:
        command.extend(["--signal", signal])

    print(
        "[pretrain-pipeline] run " + " ".join(command),
        flush=True,
    )
    subprocess.run(command, check=True, env=os.environ.copy())


def _text_parts(signal: str, row: Mapping[str, Any]) -> list[str]:
    if signal == "arithmetic":
        values: list[Any] = [
            row.get("question"),
            *(row.get("steps") if isinstance(row.get("steps"), list) else []),
            row.get("answer"),
        ]
    elif signal == "task_code":
        values = [
            row.get("task"),
            *(row.get("plan") if isinstance(row.get("plan"), list) else []),
            row.get("code"),
        ]
    elif signal in {"educational_qa_mcq_math", "educational_qa_mcq_general"}:
        values = [
            row.get("question"),
            *(row.get("choices") if isinstance(row.get("choices"), list) else []),
            row.get("explanation"),
        ]
    elif signal == "factual_restraint":
        values = [row.get("question"), row.get("safe_answer")]
    else:
        raise ValueError(f"Unsupported pretraining signal: {signal}")

    return [value for value in values if isinstance(value, str) and value]


def _estimate_tokens(
    signal: str,
    row: Mapping[str, Any],
    *,
    chars_per_token: float,
) -> int:
    text = build_public_record(signal, row)["text"]
    return max(1, math.ceil(len(text) / chars_per_token))


def _find_chars_per_token(value: Any) -> float | None:
    if isinstance(value, Mapping):
        direct = value.get("chars_per_token")
        if isinstance(direct, (int, float)) and not isinstance(direct, bool):
            if float(direct) > 0:
                return float(direct)
        for child in value.values():
            found = _find_chars_per_token(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_chars_per_token(child)
            if found is not None:
                return found
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(row)
    return rows


def _finalize(
    *,
    output_dir: Path,
    signals: Sequence[str],
    chars_per_token: float,
) -> dict[str, Any]:
    quality_dir = output_dir / "quality_accepted"
    final_dir = output_dir / "deduped"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "pretrain.jsonl"

    seen: set[str] = set()
    accepted: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    exact_dropped = 0
    per_signal: dict[str, dict[str, int]] = {}

    for signal in signals:
        rows = _read_jsonl(quality_dir / signal_to_filename(signal))
        signal_accepted = 0
        signal_tokens = 0

        for row in rows:
            public = build_public_record(signal, row)
            canonical = " ".join(public["text"].casefold().split())
            if canonical in seen:
                exact_dropped += 1
                continue

            seen.add(canonical)
            accepted.append((signal, row, public))
            signal_accepted += 1
            signal_tokens += _estimate_tokens(
                signal,
                row,
                chars_per_token=chars_per_token,
            )

        per_signal[signal] = {
            "records": signal_accepted,
            "estimated_tokens": signal_tokens,
        }

    with final_path.open("w", encoding="utf-8") as handle:
        for _, _, public in accepted:
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")

    accepted_tokens = sum(
        _estimate_tokens(signal, row, chars_per_token=chars_per_token)
        for signal, row, _ in accepted
    )

    return {
        "output": str(final_path),
        "records": len(accepted),
        "accepted_tokens": accepted_tokens,
        "exact_dropped": exact_dropped,
        "chars_per_token": chars_per_token,
        "signals": per_signal,
    }


def _write_round_config(
    *,
    base_config: Mapping[str, Any],
    output_dir: Path,
    round_number: int,
    generation_target_tokens: int,
) -> Path:
    config = dict(base_config)
    config["target_total_tokens"] = int(generation_target_tokens)

    config_dir = output_dir / "manifests" / "quality" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"round-{round_number:02d}.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _update_accepted_token_report(
    *,
    output_dir: Path,
    signal_targets: Mapping[str, int],
    finalization: Mapping[str, Any],
    generation_target_tokens: int,
    rounds: int,
    complete: bool,
    stop_reason: str | None = None,
) -> Path:
    path = output_dir / "manifests" / "accepted_token_report.json"
    report: dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            report.update(loaded)

    semantic_signals = finalization.get("signals", {})
    if not isinstance(semantic_signals, Mapping):
        semantic_signals = {}
    prior_signals = report.get("signals", {})
    if not isinstance(prior_signals, Mapping):
        prior_signals = {}

    final_signal_reports: dict[str, dict[str, Any]] = {}
    for signal, target in signal_targets.items():
        prior = prior_signals.get(signal, {})
        signal_report = dict(prior) if isinstance(prior, Mapping) else {}
        final_signal = semantic_signals.get(signal, {})
        if not isinstance(final_signal, Mapping):
            final_signal = {}
        accepted = int(final_signal.get("estimated_tokens", 0) or 0)
        signal_report.update(
            {
                "target_accepted_tokens": int(target),
                "accepted_tokens": accepted,
                "token_deficit": max(0, int(target) - accepted),
                "accepted_rows": int(final_signal.get("records", 0) or 0),
            }
        )
        final_signal_reports[signal] = signal_report

    target_tokens = sum(int(value) for value in signal_targets.values())
    accepted_tokens = sum(
        int(final_signal_reports[signal]["accepted_tokens"])
        for signal in signal_targets
    )
    deficit = sum(
        int(final_signal_reports[signal]["token_deficit"])
        for signal in signal_targets
    )
    report.update(
        {
            "schema_version": 2,
            "status": "complete" if complete else "shortfall",
            "publish_ready": bool(complete),
            "target_tokens": target_tokens,
            "target_accepted_tokens": target_tokens,
            "accepted_tokens": accepted_tokens,
            "deficit": deficit,
            "token_deficit": deficit,
            "stop_reason": (
                stop_reason
                or (
                    "accepted_token_target_reached"
                    if complete
                    else "semantic_quality_backfill_exhausted"
                )
            ),
            "signals": final_signal_reports,
            "semantic_quality": {
                "stage_order": [
                    "generation",
                    "deterministic_validation",
                    "judge",
                    "reviewer",
                    "final_exact_dedup",
                    "accepted_token_accounting",
                ],
                "final_records": int(finalization["records"]),
                "exact_dropped": int(finalization["exact_dropped"]),
                "chars_per_token": float(finalization["chars_per_token"]),
                "generation_target_tokens": int(generation_target_tokens),
                "backfill_rounds": int(rounds),
                "signals": finalization["signals"],
            },
        }
    )
    write_json(path, report)
    return path


def _quality_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "signal": args.signal,
        "judge_model": args.judge_model,
        "reviewer_model": args.reviewer_model,
        "judge_max_tokens": args.judge_max_tokens,
        "reviewer_max_tokens": args.reviewer_max_tokens,
        "judge_batch_size": args.judge_batch_size,
        "reviewer_batch_size": args.reviewer_batch_size,
        "concurrency": args.quality_concurrency,
        "stage_batch_attempts": args.stage_batch_attempts,
        "routing_mode": args.openrouter_routing_mode,
        "provider": args.openrouter_provider,
        "temperature": None,
        "top_p": None,
    }


def verify_final(config_path: str, signal: str | None = None) -> dict[str, Any]:
    cfg = load_yaml_config(config_path)
    output_dir = resolve_output_dir(cfg)
    report_path = output_dir / "manifests" / "accepted_token_report.json"
    final_path = output_dir / "deduped" / "pretrain.jsonl"

    if not report_path.exists():
        raise RuntimeError(f"Missing accepted-token report: {report_path}")
    if not final_path.exists():
        raise RuntimeError(f"Missing final pretraining dataset: {final_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    semantic = report.get("semantic_quality")
    if not isinstance(semantic, Mapping):
        raise RuntimeError(
            "Accepted-token report predates semantic quality integration; "
            "rerun pretrain-smoke or pretrain-generate."
        )

    def required_nonnegative_int(value: Any, *, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(
                f"Accepted-token report field {field!r} must be a non-negative integer"
            )
        return value

    target_value = report.get(
        "target_tokens",
        report.get("target_accepted_tokens", cfg["target_total_tokens"]),
    )
    target_tokens = required_nonnegative_int(target_value, field="target_tokens")
    accepted_tokens = required_nonnegative_int(
        report.get("accepted_tokens"), field="accepted_tokens"
    )
    deficit_value = report.get("token_deficit", report.get("deficit"))
    deficit = required_nonnegative_int(deficit_value, field="token_deficit")

    signal_reports = report.get("signals")
    semantic_signals = semantic.get("signals")
    if not isinstance(signal_reports, Mapping) or not isinstance(
        semantic_signals, Mapping
    ):
        raise RuntimeError(
            "Accepted-token report is missing per-signal completion accounting"
        )

    configured_mix = cfg.get("mix", {})
    if not isinstance(configured_mix, Mapping):
        configured_mix = {}
    expected_signals = (
        [signal]
        if signal is not None
        else [name for name in configured_mix if name in JUDGE_CRITERIA]
    )
    if not expected_signals:
        raise RuntimeError("No supported pretraining signals are configured for verification")

    missing_signals = sorted(
        name
        for name in expected_signals
        if name not in signal_reports or name not in semantic_signals
    )
    signal_deficits: dict[str, int] = {}
    for name in expected_signals:
        value = signal_reports.get(name)
        if not isinstance(value, Mapping):
            continue
        signal_deficits[name] = required_nonnegative_int(
            value.get("token_deficit"),
            field=f"signals.{name}.token_deficit",
        )

    if (
        report.get("status") != "complete"
        or report.get("publish_ready") is not True
        or accepted_tokens < target_tokens
        or deficit != 0
        or missing_signals
        or any(value != 0 for value in signal_deficits.values())
    ):
        raise RuntimeError(
            "Final post-review pretraining dataset has not reached every accepted-token allocation: "
            f"status={report.get('status')!r} publish_ready={report.get('publish_ready')!r} "
            f"accepted_tokens={accepted_tokens} target_tokens={target_tokens} "
            f"deficit={deficit} missing_signals={missing_signals} "
            f"signal_deficits={signal_deficits} "
            f"stop_reason={report.get('stop_reason')!r}"
        )

    print(
        f"[pretrain-pipeline] verified final accepted dataset "
        f"records={semantic.get('final_records', 0)} "
        f"accepted_tokens={accepted_tokens}/{target_tokens} "
        f"path={final_path}",
        flush=True,
    )
    return report


def run(args: argparse.Namespace) -> int:
    if args.verify_only:
        verify_final(args.config, signal=args.signal)
        return 0

    base_config = load_yaml_config(args.config)
    output_dir = resolve_output_dir(base_config)
    configured_mix = base_config.get("mix", {})
    if not isinstance(configured_mix, Mapping) or not configured_mix:
        raise ValueError("pretraining config must define at least one mix signal")
    if args.signal:
        if args.signal not in configured_mix:
            raise ValueError(f"requested signal {args.signal!r} is not present in the config mix")
        signals = [args.signal]
    else:
        signals = [name for name in configured_mix if name in JUDGE_CRITERIA]
    if not signals:
        raise ValueError("pretraining config contains no supported quality signals")

    signal_targets = {
        signal: _grounded_token_target(base_config, configured_mix[signal])
        for signal in signals
    }
    target_tokens = sum(signal_targets.values())
    if target_tokens < 1:
        raise ValueError("accepted-token target must be positive")

    chars_per_token = (
        _find_chars_per_token(base_config)
        or DEFAULT_CHARS_PER_TOKEN
    )
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")

    generation_target_tokens = int(base_config["target_total_tokens"])
    completed_round = 0

    for round_number in range(args.max_backfill_rounds + 1):
        completed_round = round_number
        round_config = _write_round_config(
            base_config=base_config,
            output_dir=output_dir,
            round_number=round_number,
            generation_target_tokens=generation_target_tokens,
        )

        print(
            f"[pretrain-pipeline] round={round_number} "
            f"generation_target_tokens={generation_target_tokens} "
            f"final_target_tokens={target_tokens}",
            flush=True,
        )

        _run_curate(
            config_path=round_config,
            signal=args.signal,
        )

        curate_report_path = (
            output_dir / "manifests" / "accepted_token_report.json"
        )
        curate_report: dict[str, Any] = {}
        if curate_report_path.exists():
            loaded_curate_report = json.loads(
                curate_report_path.read_text(encoding="utf-8")
            )
            if isinstance(loaded_curate_report, dict):
                curate_report = loaded_curate_report

        if round_number == 0:
            planned_target = curate_report.get("target_accepted_tokens")
            if (
                isinstance(planned_target, int)
                and not isinstance(planned_target, bool)
                and planned_target > 0
            ):
                target_tokens = planned_target
            curated_signals = curate_report.get("signals")
            if isinstance(curated_signals, Mapping):
                for signal in signals:
                    value = curated_signals.get(signal)
                    if isinstance(value, Mapping):
                        signal_target = value.get("target_accepted_tokens")
                        if (
                            isinstance(signal_target, int)
                            and not isinstance(signal_target, bool)
                            and signal_target > 0
                        ):
                            signal_targets[signal] = signal_target
                target_tokens = sum(signal_targets.values())

        inventory_exhausted = (
            curate_report.get("stop_reason") == "unique_candidate_inventory_exhausted"
        )

        run_quality(
            config_path=str(round_config),
            **_quality_kwargs(args),
        )

        finalization = _finalize(
            output_dir=output_dir,
            signals=signals,
            chars_per_token=chars_per_token,
        )
        accepted_tokens = int(finalization["accepted_tokens"])
        final_signal_values = finalization.get("signals", {})
        signal_deficits = {
            signal: max(
                0,
                signal_targets[signal]
                - int(final_signal_values.get(signal, {}).get("estimated_tokens", 0) or 0),
            )
            for signal in signals
        }
        allocation_deficit = sum(signal_deficits.values())
        global_token_deficit = max(0, target_tokens - accepted_tokens)
        complete = allocation_deficit == 0

        report_path = _update_accepted_token_report(
            output_dir=output_dir,
            signal_targets=signal_targets,
            finalization=finalization,
            generation_target_tokens=generation_target_tokens,
            rounds=round_number,
            complete=complete,
            stop_reason=(
                "unique_candidate_inventory_exhausted"
                if inventory_exhausted and not complete
                else None
            ),
        )

        print(
            f"[pretrain-pipeline] round={round_number} "
            f"final_records={finalization['records']} "
            f"accepted_tokens={accepted_tokens}/{target_tokens} "
            f"global_token_deficit={global_token_deficit} "
            f"allocation_deficit={allocation_deficit} "
            f"signal_deficits={signal_deficits} "
            f"exact_dropped={finalization['exact_dropped']}",
            flush=True,
        )

        if complete:
            print(
                f"[pretrain-pipeline] complete report={report_path} "
                f"dataset={finalization['output']}",
                flush=True,
            )
            return 0

        if inventory_exhausted:
            raise RuntimeError(
                "Pretraining candidate inventory exhausted before every post-review "
                "accepted-token allocation was reached: "
                f"accepted_tokens={accepted_tokens}/{target_tokens} "
                f"report={report_path} dataset={finalization['output']}"
            )

        if round_number >= args.max_backfill_rounds:
            break

        deficit = allocation_deficit
        survival = min(
            1.0,
            max(
                0.05,
                accepted_tokens / max(1, generation_target_tokens),
            ),
        )
        additional_generation_tokens = math.ceil(
            (deficit / survival) * args.backfill_headroom
        )
        generation_target_tokens += max(1, additional_generation_tokens)

        print(
            f"[pretrain-pipeline] backfill deficit={deficit} "
            f"observed_survival={survival:.4f} "
            f"next_generation_target_tokens={generation_target_tokens}",
            flush=True,
        )

    report_path = output_dir / "manifests" / "accepted_token_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    signal_reports = report.get("signals", {}) if isinstance(report, Mapping) else {}
    remaining = {
        signal: int(signal_reports.get(signal, {}).get("token_deficit", -1))
        for signal in signals
        if isinstance(signal_reports, Mapping)
    }
    raise RuntimeError(
        "Pretraining semantic-quality backfill exhausted before every per-signal "
        f"accepted-token allocation was reached after {completed_round} backfill round(s). "
        f"global_accepted_tokens={report.get('accepted_tokens')} "
        f"global_target_tokens={report.get('target_tokens')} "
        f"signal_deficits={remaining}. "
        "Accepted output and quality reports were preserved."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production pretraining pipeline: grounded generation, "
            "deterministic validation, judge, reviewer, final exact dedup, "
            "and accepted-token backfill."
        )
    )
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--signal", choices=[*JUDGE_CRITERIA], default=None)
    parser.add_argument("--verify-only", action="store_true")

    parser.add_argument("--judge-model", default="google/gemma-4-31b-it")
    parser.add_argument("--reviewer-model", default="openai/gpt-5.6-luna-pro")
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--reviewer-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-batch-size", type=int, default=10)
    parser.add_argument("--reviewer-batch-size", type=int, default=10)
    parser.add_argument("--quality-concurrency", type=int, default=8)
    parser.add_argument("--stage-batch-attempts", type=int, default=3)
    parser.add_argument("--max-backfill-rounds", type=int, default=4)
    parser.add_argument("--backfill-headroom", type=float, default=1.05)

    parser.add_argument(
        "--openrouter-routing-mode",
        choices=("auto", "prefer", "strict"),
        default="auto",
    )
    parser.add_argument("--openrouter-provider", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for name in (
        "judge_max_tokens",
        "reviewer_max_tokens",
        "judge_batch_size",
        "reviewer_batch_size",
        "quality_concurrency",
        "stage_batch_attempts",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.max_backfill_rounds < 0:
        parser.error("--max-backfill-rounds must be non-negative")
    if args.backfill_headroom < 1.0:
        parser.error("--backfill-headroom must be at least 1.0")

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
