from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.pretrain.record_quality import (
    SIGNAL_FROM_FILE,
    iter_jsonl,
    signal_to_filename,
)
from slm_synth.runtime import (
    build_backend,
    chunked,
    run_model_stage_with_isolation,
    split_sequence_batch,
    write_json,
    write_jsonl,
)


JUDGE_CRITERIA: dict[str, str] = {
    "arithmetic": (
        "A strong arithmetic pretraining record must be mathematically correct, internally consistent, "
        "clear, natural, and useful as training text. Reject awkward or obviously synthetic/template-like "
        "phrasing, contradictory wording, misleading context, incoherent worked steps, or content whose "
        "surface form is materially lower quality than the grounded arithmetic itself."
    ),
    "task_code": (
        "A strong task_code pretraining record must faithfully describe the supplied code, give a useful "
        "and coherent plan, and avoid claiming behavior the code does not implement. Reject incorrect or "
        "misleading task descriptions, plans that contradict the code, malformed/incomplete code, or "
        "unnatural low-quality synthetic text."
    ),
    "educational_qa_mcq_math": (
        "A strong math MCQ pretraining record must be mathematically correct, coherent, self-contained, "
        "natural, and educationally useful. Reject contradictions, misleading explanations, answer leakage, "
        "or awkward/template-like text that materially lowers training quality."
    ),
    "educational_qa_mcq_general": (
        "A strong general MCQ pretraining record must be faithful to the supplied evidence/question/choices, "
        "have a correct and useful explanation, and read naturally. Reject unsupported explanations, "
        "contradictions, or low-quality/template-like prose."
    ),
    "factual_restraint": (
        "A strong factual-restraint pretraining record must respond appropriately to the question, avoid "
        "inventing facts, reflect the intended uncertainty/restraint behavior, and read naturally. Reject "
        "hallucinated claims, inappropriate refusal/compliance, contradictions, or low-quality/template-like prose."
    ),
}


def _signals_from_args(validated_dir: Path, signal: str | None) -> list[str]:
    if signal:
        if signal not in JUDGE_CRITERIA:
            raise ValueError(f"Unsupported pretraining signal: {signal}")
        return [signal]
    signals: list[str] = []
    for path in sorted(validated_dir.glob("*.jsonl")):
        candidate = SIGNAL_FROM_FILE.get(path.name, path.stem)
        if candidate in JUDGE_CRITERIA:
            signals.append(candidate)
    return signals


def _record_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_validated_rows(validated_dir: Path, signal: str) -> list[dict[str, Any]]:
    path = validated_dir / signal_to_filename(signal)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line_number, row, issues in iter_jsonl(path):
        if issues or row is None:
            raise ValueError(
                f"Validated pretraining input is not valid JSON at {path}:{line_number}: {issues}"
            )
        rows.append(
            {
                "artifact_id": f"{signal}:{line_number:09d}",
                "signal": signal,
                "record_sha256": _record_sha256(row),
                "record": row,
            }
        )
    return rows


def _load_decision_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid cached quality decision at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"Cached quality decision must be an object at {path}:{line_number}"
            )
        artifact_id = row.get("artifact_id")
        record_sha256 = row.get("record_sha256")
        if isinstance(artifact_id, str) and isinstance(record_sha256, str):
            cache[artifact_id] = row
    return cache


def judge_schema(expected_ids: Sequence[str]) -> dict[str, Any]:
    if not expected_ids:
        raise ValueError("judge schema requires at least one artifact id")

    decision = {
        "type": "object",
        "properties": {
            "assessable": {"type": "boolean"},
            "quality_valid": {"type": "boolean"},
            "signal_aligned": {"type": "boolean"},
            "natural_and_useful": {"type": "boolean"},
            "reason": {"type": "string", "minLength": 1},
        },
        "required": [
            "assessable",
            "quality_valid",
            "signal_aligned",
            "natural_and_useful",
            "reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {artifact_id: decision for artifact_id in expected_ids},
                "required": list(expected_ids),
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def reviewer_schema(expected_ids: Sequence[str]) -> dict[str, Any]:
    if not expected_ids:
        raise ValueError("reviewer schema requires at least one artifact id")

    decision = {
        "type": "object",
        "properties": {
            "reviewed": {"type": "boolean"},
            "reviewer_agreed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["reviewed", "reviewer_agreed", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {artifact_id: decision for artifact_id in expected_ids},
                "required": list(expected_ids),
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _call_structured(
    backend: Any,
    *,
    prompt: str,
    schema: Mapping[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    with_metadata = getattr(backend, "generate_structured_object_with_metadata", None)
    if with_metadata is not None:
        envelope = with_metadata(
            prompt=prompt,
            schema=dict(schema),
            schema_name=schema_name,
        )
        if not isinstance(envelope, Mapping):
            raise TypeError("structured backend returned a non-object envelope")
        data = envelope.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("structured backend response missing object data")
        return dict(data)

    method = getattr(backend, "generate_structured_object", None)
    if method is None:
        raise TypeError("backend does not implement a structured-object generation method")
    result = method(
        prompt=prompt,
        schema=dict(schema),
        schema_name=schema_name,
    )
    if not isinstance(result, Mapping):
        raise TypeError("structured backend returned a non-object response")
    return dict(result)


def _parse_judge(
    parsed: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    decisions = parsed.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("judge response must contain decisions object")
    if set(decisions) != set(expected_ids):
        raise ValueError("judge response has missing or unexpected artifact ids")

    normalized: dict[str, dict[str, Any]] = {}
    for artifact_id in expected_ids:
        raw = decisions[artifact_id]
        if not isinstance(raw, Mapping):
            raise ValueError(f"judge decision must be an object for {artifact_id}")
        assessable = raw.get("assessable")
        quality_valid = raw.get("quality_valid")
        signal_aligned = raw.get("signal_aligned")
        natural_and_useful = raw.get("natural_and_useful")
        reason = raw.get("reason")
        if not all(
            isinstance(value, bool)
            for value in (
                assessable,
                quality_valid,
                signal_aligned,
                natural_and_useful,
            )
        ):
            raise ValueError(f"judge booleans are invalid for {artifact_id}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"judge reason missing for {artifact_id}")

        accepted = bool(
            assessable
            and quality_valid
            and signal_aligned
            and natural_and_useful
        )
        normalized[artifact_id] = {
            "artifact_id": artifact_id,
            "assessable": assessable,
            "quality_valid": quality_valid,
            "signal_aligned": signal_aligned,
            "natural_and_useful": natural_and_useful,
            "judge_reason": reason.strip(),
            "judge_accepted": accepted,
        }
    return normalized


def _parse_reviewer(
    parsed: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    decisions = parsed.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("reviewer response must contain decisions object")
    if set(decisions) != set(expected_ids):
        raise ValueError("reviewer response has missing or unexpected artifact ids")

    normalized: dict[str, dict[str, Any]] = {}
    for artifact_id in expected_ids:
        raw = decisions[artifact_id]
        if not isinstance(raw, Mapping):
            raise ValueError(f"reviewer decision must be an object for {artifact_id}")
        reviewed = raw.get("reviewed")
        agreed = raw.get("reviewer_agreed")
        reason = raw.get("reason")
        if not isinstance(reviewed, bool) or not isinstance(agreed, bool):
            raise ValueError(f"reviewer booleans are invalid for {artifact_id}")
        if not isinstance(reason, str):
            raise ValueError(f"reviewer reason must be a string for {artifact_id}")
        accepted = bool(reviewed and agreed)
        normalized[artifact_id] = {
            "artifact_id": artifact_id,
            "reviewed": reviewed,
            "reviewer_agreed": agreed,
            "reviewer_reason": reason.strip(),
            "accepted": accepted,
        }
    return normalized


def _judge_batch(
    *,
    signal: str,
    rows: Sequence[Mapping[str, Any]],
    backend: Any,
) -> dict[str, dict[str, Any]]:
    expected_ids = [str(row["artifact_id"]) for row in rows]
    payload = [
        {
            "artifact_id": row["artifact_id"],
            "record": row["record"],
        }
        for row in rows
    ]
    prompt = (
        "You are the quality judge for synthetic PRETRAINING records.\n\n"
        f"Signal: {signal}\n\n"
        f"{JUDGE_CRITERIA[signal]}\n\n"
        "Evaluate each record independently.\n"
        "- assessable: false only when the record cannot be meaningfully evaluated.\n"
        "- quality_valid: true only when the record is substantively correct and internally coherent.\n"
        "- signal_aligned: true only when it teaches the intended signal rather than some different behavior.\n"
        "- natural_and_useful: true only when the text is natural enough and useful enough to keep as pretraining data.\n"
        "Do not approve a record merely because its structure is valid.\n"
        "Return one decision per artifact_id and preserve every artifact_id exactly.\n\n"
        "RECORDS:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    parsed = _call_structured(
        backend,
        prompt=prompt,
        schema=judge_schema(expected_ids),
        schema_name=f"pretrain_judge_{signal}_{len(rows)}",
    )
    return _parse_judge(parsed, expected_ids)


def _reviewer_batch(
    *,
    signal: str,
    rows: Sequence[Mapping[str, Any]],
    judge_decisions: Mapping[str, Mapping[str, Any]],
    backend: Any,
) -> dict[str, dict[str, Any]]:
    expected_ids = [str(row["artifact_id"]) for row in rows]
    payload = [
        {
            "artifact_id": row["artifact_id"],
            "record": row["record"],
            "judge": judge_decisions[row["artifact_id"]],
        }
        for row in rows
    ]
    prompt = (
        "You are the reviewer for synthetic PRETRAINING records that a prior judge accepted.\n\n"
        f"Signal: {signal}\n\n"
        "Your role is not to rescue rows the judge rejected. Review only the accepted rows below.\n"
        "Independently inspect whether each accepted row is actually suitable high-quality pretraining material.\n"
        "Look for mistakes the judge may have missed: factual or mathematical errors, task/code mismatches, "
        "contradictions, unsupported claims, malformed or incomplete content, misleading wording, or clearly "
        "low-quality synthetic/template text.\n\n"
        "- reviewed must be true when you can meaningfully inspect the record.\n"
        "- reviewer_agreed=true means you agree with the judge's acceptance.\n"
        "- reviewer_agreed=false means the judge accepted a row that should not survive.\n"
        "Do not reject merely because a row is imperfect or stylistically ordinary. The target is high quality, "
        "not perfection or 100% rejection precision.\n"
        "Preserve artifact_id exactly and return one decision for every supplied row.\n\n"
        "ACCEPTED RECORDS TO REVIEW:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    parsed = _call_structured(
        backend,
        prompt=prompt,
        schema=reviewer_schema(expected_ids),
        schema_name=f"pretrain_reviewer_{signal}_{len(rows)}",
    )
    return _parse_reviewer(parsed, expected_ids)


def run_quality(
    *,
    config_path: str,
    signal: str | None,
    judge_model: str,
    reviewer_model: str,
    judge_max_tokens: int,
    reviewer_max_tokens: int,
    judge_batch_size: int,
    reviewer_batch_size: int,
    concurrency: int,
    stage_batch_attempts: int,
    routing_mode: str,
    provider: str | None,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    cfg = load_yaml_config(config_path)
    out = resolve_output_dir(cfg)
    validated_dir = out / "validated"
    accepted_dir = out / "quality_accepted"
    manifest_dir = out / "manifests" / "quality"
    rejected_dir = out / "rejected"

    accepted_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    signals = _signals_from_args(validated_dir, signal)
    if not signals:
        raise ValueError(f"No validated pretraining signals found under {validated_dir}")

    judge_backend = build_backend(
        model=judge_model,
        max_tokens=judge_max_tokens,
        concurrency=concurrency,
        routing_mode=routing_mode,
        provider=provider,
        # Optional sampling controls remain unset so any qualified model/provider
        # route can be used for the judge role.
        temperature=None,
        top_p=None,
    )
    reviewer_backend = build_backend(
        model=reviewer_model,
        max_tokens=reviewer_max_tokens,
        concurrency=concurrency,
        routing_mode=routing_mode,
        provider=provider,
        # Luna reviewer must leave these unset by default; forcing temperature=0
        # caused OpenRouter endpoint-selection failures in the finalized probe.
        temperature=temperature,
        top_p=top_p,
    )

    totals = {
        "validated": 0,
        "judge_accepted": 0,
        "judge_rejected": 0,
        "judge_unrecoverable": 0,
        "reviewer_accepted": 0,
        "reviewer_rejected": 0,
        "reviewer_unrecoverable": 0,
    }
    signal_summaries: dict[str, Any] = {}

    for current_signal in signals:
        rows = _load_validated_rows(validated_dir, current_signal)
        totals["validated"] += len(rows)

        rows_by_id = {row["artifact_id"]: row for row in rows}
        judge_cache_path = manifest_dir / f"{current_signal}.judge.jsonl"
        cached_judge = _load_decision_cache(judge_cache_path)
        judge_decisions: dict[str, dict[str, Any]] = {
            artifact_id: decision
            for artifact_id, decision in cached_judge.items()
            if artifact_id in rows_by_id
            and decision.get("record_sha256")
            == rows_by_id[artifact_id]["record_sha256"]
        }
        uncached_judge_rows = [
            row for row in rows if row["artifact_id"] not in judge_decisions
        ]

        judge_batches = chunked(uncached_judge_rows, judge_batch_size)
        judge_results, judge_failures = run_model_stage_with_isolation(
            stage=f"pretrain-judge:{current_signal}",
            batches=judge_batches,
            item_count=len,
            worker=lambda batch, s=current_signal: _judge_batch(
                signal=s,
                rows=batch,
                backend=judge_backend,
            ),
            split_batch=split_sequence_batch,
            concurrency=concurrency,
            batch_size_display=judge_batch_size,
            max_attempts=stage_batch_attempts,
        ) if judge_batches else ([], [])

        for result in judge_results:
            judge_decisions.update(result)
        for artifact_id, decision in judge_decisions.items():
            decision["record_sha256"] = rows_by_id[artifact_id]["record_sha256"]

        missing_judge = [
            artifact_id
            for artifact_id in rows_by_id
            if artifact_id not in judge_decisions
        ]
        judge_accepted_ids = [
            artifact_id
            for artifact_id, decision in judge_decisions.items()
            if decision["judge_accepted"]
        ]
        judge_rejected_ids = [
            artifact_id
            for artifact_id, decision in judge_decisions.items()
            if not decision["judge_accepted"]
        ]

        review_rows = [rows_by_id[artifact_id] for artifact_id in judge_accepted_ids]
        reviewer_cache_path = manifest_dir / f"{current_signal}.reviewer.jsonl"
        cached_reviewer = _load_decision_cache(reviewer_cache_path)
        reviewer_decisions: dict[str, dict[str, Any]] = {
            artifact_id: decision
            for artifact_id, decision in cached_reviewer.items()
            if artifact_id in rows_by_id
            and artifact_id in judge_accepted_ids
            and decision.get("record_sha256")
            == rows_by_id[artifact_id]["record_sha256"]
        }
        uncached_review_rows = [
            row for row in review_rows
            if row["artifact_id"] not in reviewer_decisions
        ]

        reviewer_batches = chunked(uncached_review_rows, reviewer_batch_size)
        reviewer_results, reviewer_failures = run_model_stage_with_isolation(
            stage=f"pretrain-reviewer:{current_signal}",
            batches=reviewer_batches,
            item_count=len,
            worker=lambda batch, s=current_signal: _reviewer_batch(
                signal=s,
                rows=batch,
                judge_decisions=judge_decisions,
                backend=reviewer_backend,
            ),
            split_batch=split_sequence_batch,
            concurrency=concurrency,
            batch_size_display=reviewer_batch_size,
            max_attempts=stage_batch_attempts,
        ) if reviewer_batches else ([], [])

        for result in reviewer_results:
            reviewer_decisions.update(result)
        for artifact_id, decision in reviewer_decisions.items():
            decision["record_sha256"] = rows_by_id[artifact_id]["record_sha256"]

        missing_reviewer = [
            artifact_id
            for artifact_id in judge_accepted_ids
            if artifact_id not in reviewer_decisions
        ]
        final_ids = [
            artifact_id
            for artifact_id in judge_accepted_ids
            if artifact_id in reviewer_decisions
            and reviewer_decisions[artifact_id]["accepted"]
        ]
        reviewer_rejected_ids = [
            artifact_id
            for artifact_id in judge_accepted_ids
            if artifact_id in reviewer_decisions
            and not reviewer_decisions[artifact_id]["accepted"]
        ]

        final_records = [rows_by_id[artifact_id]["record"] for artifact_id in final_ids]
        write_jsonl(
            accepted_dir / signal_to_filename(current_signal),
            final_records,
        )
        write_jsonl(
            manifest_dir / f"{current_signal}.judge.jsonl",
            [judge_decisions[key] for key in sorted(judge_decisions)],
        )
        write_jsonl(
            manifest_dir / f"{current_signal}.reviewer.jsonl",
            [reviewer_decisions[key] for key in sorted(reviewer_decisions)],
        )
        write_jsonl(
            manifest_dir / f"{current_signal}.failures.jsonl",
            [*judge_failures, *reviewer_failures],
        )

        semantic_rejections = []
        for artifact_id in judge_rejected_ids:
            semantic_rejections.append(
                {
                    "stage": "judge",
                    "artifact_id": artifact_id,
                    "signal": current_signal,
                    "record": rows_by_id[artifact_id]["record"],
                    "decision": judge_decisions[artifact_id],
                }
            )
        for artifact_id in missing_judge:
            semantic_rejections.append(
                {
                    "stage": "judge_unrecoverable",
                    "artifact_id": artifact_id,
                    "signal": current_signal,
                    "record": rows_by_id[artifact_id]["record"],
                }
            )
        for artifact_id in reviewer_rejected_ids:
            semantic_rejections.append(
                {
                    "stage": "reviewer",
                    "artifact_id": artifact_id,
                    "signal": current_signal,
                    "record": rows_by_id[artifact_id]["record"],
                    "judge": judge_decisions[artifact_id],
                    "review": reviewer_decisions[artifact_id],
                }
            )
        for artifact_id in missing_reviewer:
            semantic_rejections.append(
                {
                    "stage": "reviewer_unrecoverable",
                    "artifact_id": artifact_id,
                    "signal": current_signal,
                    "record": rows_by_id[artifact_id]["record"],
                    "judge": judge_decisions[artifact_id],
                }
            )
        write_jsonl(
            rejected_dir / f"{current_signal}.semantic.jsonl",
            semantic_rejections,
        )

        summary = {
            "signal": current_signal,
            "validated": len(rows),
            "judge_accepted": len(judge_accepted_ids),
            "judge_rejected": len(judge_rejected_ids),
            "judge_unrecoverable": len(missing_judge),
            "reviewer_accepted": len(final_ids),
            "reviewer_rejected": len(reviewer_rejected_ids),
            "reviewer_unrecoverable": len(missing_reviewer),
            "accepted_output": str(
                accepted_dir / signal_to_filename(current_signal)
            ),
        }
        signal_summaries[current_signal] = summary
        write_json(manifest_dir / f"{current_signal}.summary.json", summary)

        totals["judge_accepted"] += len(judge_accepted_ids)
        totals["judge_rejected"] += len(judge_rejected_ids)
        totals["judge_unrecoverable"] += len(missing_judge)
        totals["reviewer_accepted"] += len(final_ids)
        totals["reviewer_rejected"] += len(reviewer_rejected_ids)
        totals["reviewer_unrecoverable"] += len(missing_reviewer)

        print(
            f"[pretrain-quality] {current_signal}: "
            f"validated={len(rows)} "
            f"judge_accepted={len(judge_accepted_ids)} "
            f"reviewer_accepted={len(final_ids)}",
            flush=True,
        )

    report = {
        "models": {
            "judge": judge_model,
            "reviewer": reviewer_model,
        },
        "signals": signal_summaries,
        "totals": totals,
    }
    write_json(manifest_dir / "summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the finalized semantic quality gates on validated pretraining records."
    )
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--signal", default=None, choices=[*JUDGE_CRITERIA])
    parser.add_argument("--judge-model", default="google/gemma-4-31b-it")
    parser.add_argument("--reviewer-model", default="openai/gpt-5.6-luna-pro")
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--reviewer-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-batch-size", type=int, default=10)
    parser.add_argument("--reviewer-batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--stage-batch-attempts", type=int, default=3)
    parser.add_argument(
        "--openrouter-routing-mode",
        default="auto",
        choices=("auto", "prefer", "strict"),
    )
    parser.add_argument("--openrouter-provider", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in (
        "judge_max_tokens",
        "reviewer_max_tokens",
        "judge_batch_size",
        "reviewer_batch_size",
        "concurrency",
        "stage_batch_attempts",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    report = run_quality(
        config_path=args.config,
        signal=args.signal,
        judge_model=args.judge_model,
        reviewer_model=args.reviewer_model,
        judge_max_tokens=args.judge_max_tokens,
        reviewer_max_tokens=args.reviewer_max_tokens,
        judge_batch_size=args.judge_batch_size,
        reviewer_batch_size=args.reviewer_batch_size,
        concurrency=args.concurrency,
        stage_batch_attempts=args.stage_batch_attempts,
        routing_mode=args.openrouter_routing_mode,
        provider=args.openrouter_provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
