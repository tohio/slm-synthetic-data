"""Backfill rows removed by Distillation-SFT response adjudication."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from slm_synth.accepted_target import accepted_target_metadata
from slm_synth.distillation_sft.orchestration import (
    BackendFactory,
    generate_prompt_spec_multi_signal_run,
)
from slm_synth.distillation_sft.prompt_quality import normalize_prompt_text
from slm_synth.distillation_sft.schema import validate_public_row
from slm_synth.telemetry import aggregate_llm_telemetry

_INDEXED_ID_RE = re.compile(r"^(?P<signal>[a-z][a-z0-9_]*)-(?P<index>\d+)$")


def backfill_adjudicated_run(
    *,
    run_dir: str | Path,
    max_tokens: int,
    batch_size: int,
    concurrency: int,
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_initial_in_flight: int = 8,
    adaptive_initial_batch_size: int = 4,
    adaptive_batch_increase_successes: int = 4,
    max_backfill_rounds: int = 2,
    openrouter_routing_mode: str | None = None,
    openrouter_provider: str | None = None,
    backend_factory: BackendFactory | None = None,
) -> dict[str, Any]:
    """Generate exactly the per-signal deficits and merge them transactionally."""
    root = Path(run_dir)
    dataset_dir = root / "datasets"
    manifest_path = _discover_run_manifest(root)
    manifest = _read_json_object(manifest_path, label="run manifest")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"distillation run manifest has no datasets: {manifest_path}")

    targets: dict[str, int] = {}
    dataset_paths: dict[str, Path] = {}
    signal_manifest_paths: dict[str, Path] = {}
    existing_rows: dict[str, list[dict[str, Any]]] = {}
    for entry in datasets:
        if not isinstance(entry, Mapping):
            raise ValueError("distillation run manifest dataset entries must be objects")
        signal = entry.get("signal")
        target = entry.get("row_count")
        if not isinstance(signal, str) or not isinstance(target, int) or target < 0:
            raise ValueError("distillation run manifest contains an invalid signal target")
        dataset_path = dataset_dir / f"{signal}.jsonl"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"adjudicated dataset does not exist: {dataset_path}")
        rows = _read_public_jsonl(dataset_path)
        if len(rows) > target:
            raise ValueError(
                f"adjudicated {signal} dataset exceeds its manifest target: "
                f"dataset={len(rows)} target={target}"
            )
        targets[signal] = target
        dataset_paths[signal] = dataset_path
        existing_rows[signal] = rows
        signal_manifest_paths[signal] = _resolve_manifest_path(root, entry.get("manifest_path"))

    deficits = {
        signal: targets[signal] - len(existing_rows[signal])
        for signal in sorted(targets)
        if len(existing_rows[signal]) < targets[signal]
    }
    if not deficits:
        return {
            "generation_run": manifest.get("generation_run"),
            "added_rows": 0,
            "rows": sum(len(rows) for rows in existing_rows.values()),
            "signals": {},
        }

    generation_run = manifest.get("generation_run")
    teacher_model = manifest.get("teacher_model")
    if not isinstance(generation_run, str) or not generation_run:
        raise ValueError("distillation run manifest is missing generation_run")
    if not isinstance(teacher_model, str) or not teacher_model:
        raise ValueError("distillation run manifest is missing teacher_model")
    metadata = manifest.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    previous_backfills = metadata.get("adjudication_backfills", [])
    if not isinstance(previous_backfills, list):
        raise ValueError("run manifest adjudication_backfills must be a list")
    round_number = len(previous_backfills) + 1
    backfill_run = f"{generation_run}-adjudication-backfill-{round_number:03d}"
    staging_root = root / "retries" / "adjudication_backfill" / backfill_run

    rejected_indexes = _load_rejected_indexes(root / "rejected")
    for signal, deficit in deficits.items():
        quarantined = len(rejected_indexes.get(signal, set()))
        if quarantined < deficit:
            raise ValueError(
                f"cannot backfill unexplained {signal} deficit: missing_rows={deficit} "
                f"quarantined_adjudication_rows={quarantined}"
            )
    staged_rows: dict[str, list[dict[str, Any]]] = {}
    staged_manifests: dict[str, Path] = {}
    starts: dict[str, int] = {}
    for signal, deficit in deficits.items():
        starts[signal] = _next_source_index(
            signal=signal,
            target=targets[signal],
            rows=existing_rows[signal],
            rejected_indexes=rejected_indexes,
        )
        signal_root = staging_root / signal
        result = generate_prompt_spec_multi_signal_run(
            signals=[signal],
            count_per_signal=deficit,
            output_dir=signal_root / "datasets",
            manifest_dir=signal_root / "manifests",
            teacher_model=teacher_model,
            generation_run=backfill_run,
            max_tokens=max_tokens,
            start_index=starts[signal],
            temperature=temperature,
            top_p=top_p,
            request_timeout=request_timeout,
            max_request_retries=max_request_retries,
            max_retryable_request_attempts=max_retryable_request_attempts,
            retry_max_elapsed_seconds=retry_max_elapsed_seconds,
            adaptive_maximum_in_flight=concurrency,
            adaptive_initial_in_flight=adaptive_initial_in_flight,
            openrouter_routing_mode=openrouter_routing_mode,
            openrouter_provider=openrouter_provider,
            adaptive_initial_batch_size=adaptive_initial_batch_size,
            adaptive_batch_increase_successes=adaptive_batch_increase_successes,
            batch_size=batch_size,
            concurrency=concurrency,
            max_backfill_rounds=max_backfill_rounds,
            backend_factory=backend_factory,
        )
        staged_rows[signal] = _read_public_jsonl(result.results[0].dataset_path)
        staged_manifests[signal] = result.results[0].manifest_path
        if len(staged_rows[signal]) != deficit:
            raise ValueError(
                f"adjudication backfill for {signal} was underfilled: "
                f"expected={deficit} generated={len(staged_rows[signal])}"
            )

    merged_rows = {
        signal: existing_rows[signal] + staged_rows.get(signal, [])
        for signal in sorted(existing_rows)
    }
    _validate_merged_rows(merged_rows, targets=targets)

    for signal, rows in merged_rows.items():
        _write_public_jsonl_atomically(dataset_paths[signal], rows)
    backfill_record = {
        "generation_run": backfill_run,
        "added_rows": sum(deficits.values()),
        "signals": {
            signal: {
                "added_rows": deficits[signal],
                "start_index": starts[signal],
                "staged_manifest": str(staged_manifests[signal]),
            }
            for signal in sorted(deficits)
        },
    }
    for signal in deficits:
        _update_signal_manifest(
            signal_manifest_paths[signal],
            row_count=len(merged_rows[signal]),
            target=targets[signal],
            backfill_record=backfill_record["signals"][signal],
            staged_manifest_path=staged_manifests[signal],
        )
    _update_run_manifest(
        manifest_path,
        manifest=manifest,
        row_counts={signal: len(rows) for signal, rows in merged_rows.items()},
        backfill_record=backfill_record,
        signal_manifest_paths=signal_manifest_paths,
    )
    return {
        "generation_run": generation_run,
        "backfill_run": backfill_run,
        "added_rows": sum(deficits.values()),
        "rows": sum(len(rows) for rows in merged_rows.values()),
        "signals": backfill_record["signals"],
    }


def _discover_run_manifest(root: Path) -> Path:
    expected = root / "manifests" / f"{root.name}.manifest.json"
    if expected.is_file():
        return expected
    candidates = []
    for path in sorted((root / "manifests").glob("*.manifest.json")):
        payload = _read_json_object(path, label="manifest")
        if isinstance(payload.get("datasets"), list):
            candidates.append(path)
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one run manifest under {root / 'manifests'}")
    return candidates[0]


def _resolve_manifest_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("run manifest dataset entry is missing manifest_path")
    path = Path(value)
    if path.is_file():
        return path
    local = root / "manifests" / path.name
    if not local.is_file():
        raise FileNotFoundError(f"signal manifest does not exist: {value}")
    return local


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _read_public_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
            rows.append(validate_public_row(value))
    return rows


def _load_rejected_indexes(rejected_dir: Path) -> dict[str, set[int]]:
    indexes: dict[str, set[int]] = {}
    if not rejected_dir.is_dir():
        return indexes
    for path in sorted(rejected_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            row = value.get("row") if isinstance(value, Mapping) else None
            if not isinstance(row, Mapping):
                continue
            match = _INDEXED_ID_RE.fullmatch(str(row.get("id", "")))
            if match:
                indexes.setdefault(match.group("signal"), set()).add(int(match.group("index")))
    return indexes


def _next_source_index(
    *,
    signal: str,
    target: int,
    rows: list[Mapping[str, Any]],
    rejected_indexes: Mapping[str, set[int]],
) -> int:
    indexes = set(rejected_indexes.get(signal, set()))
    for row in rows:
        match = _INDEXED_ID_RE.fullmatch(str(row.get("id", "")))
        if match and match.group("signal") == signal:
            indexes.add(int(match.group("index")))
    return max([target, *indexes], default=target) + 1


def _validate_merged_rows(
    rows_by_signal: Mapping[str, list[dict[str, Any]]],
    *,
    targets: Mapping[str, int],
) -> None:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for signal, rows in rows_by_signal.items():
        if len(rows) != targets[signal]:
            raise ValueError(
                f"merged adjudication backfill count mismatch for {signal}: "
                f"rows={len(rows)} target={targets[signal]}"
            )
        for row in rows:
            row_id = row["id"]
            if row_id in seen_ids:
                raise ValueError(f"merged adjudication backfill contains duplicate row id: {row_id}")
            seen_ids.add(row_id)
            prompt = normalize_prompt_text(row["prompt"])
            if prompt in seen_prompts:
                raise ValueError("merged adjudication backfill contains duplicate prompt text")
            seen_prompts.add(prompt)


def _update_signal_manifest(
    path: Path,
    *,
    row_count: int,
    target: int,
    backfill_record: Mapping[str, Any],
    staged_manifest_path: Path,
) -> None:
    manifest = _read_json_object(path, label="signal manifest")
    metadata = manifest.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    records = metadata.get("adjudication_backfills", [])
    records = list(records) if isinstance(records, list) else []
    records.append(dict(backfill_record))
    staged_manifest = _read_json_object(staged_manifest_path, label="staged signal manifest")
    staged_metadata = staged_manifest.get("metadata")
    staged_metadata = dict(staged_metadata) if isinstance(staged_metadata, Mapping) else {}
    attempted = max(int(metadata.get("planned_prompt_rows", row_count)), row_count) + int(
        staged_metadata.get("planned_prompt_rows", backfill_record["added_rows"])
    )
    rejection_reasons = dict(metadata.get("rejection_reasons", {}))
    rejection_reasons["repeated_response_cluster_adjudication"] = (
        int(rejection_reasons.get("repeated_response_cluster_adjudication", 0))
        + int(backfill_record["added_rows"])
    )
    staged_reasons = staged_metadata.get("rejection_reasons", {})
    if isinstance(staged_reasons, Mapping):
        for reason, count in staged_reasons.items():
            if isinstance(reason, str) and isinstance(count, int):
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count
    telemetry_items = []
    for source in (metadata, staged_metadata):
        telemetry = source.get("llm_telemetry")
        if isinstance(telemetry, Mapping):
            telemetry_items.append(dict(telemetry))
    metadata.update(
        {
            "accepted_rows": row_count,
            "target_prompt_rows": target,
            "planned_prompt_rows": attempted,
            "rejected_rows": attempted - row_count,
            "rejection_reasons": rejection_reasons,
            "response_quality": {
                "checked_rows": attempted,
                "accepted_rows": row_count,
                "rejected_rows": attempted - row_count,
                "rejection_reasons": rejection_reasons,
            },
            "llm_telemetry": aggregate_llm_telemetry(telemetry_items),
            "adjudication_backfills": records,
            **accepted_target_metadata(
                unit="rows",
                target_count=target,
                accepted_count=row_count,
                attempted_count=attempted,
                max_backfill_rounds=int(metadata.get("max_backfill_rounds", 0)),
                backfill_rounds=int(metadata.get("backfill_rounds", 0)),
            ),
        }
    )
    manifest["row_count"] = row_count
    manifest["metadata"] = metadata
    _write_json_atomically(path, manifest)


def _update_run_manifest(
    path: Path,
    *,
    manifest: dict[str, Any],
    row_counts: Mapping[str, int],
    backfill_record: Mapping[str, Any],
    signal_manifest_paths: Mapping[str, Path],
) -> None:
    datasets = []
    for raw_entry in manifest["datasets"]:
        entry = dict(raw_entry)
        entry["row_count"] = row_counts[entry["signal"]]
        datasets.append(entry)
    total_rows = sum(row_counts.values())
    metadata = manifest.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    records = metadata.get("adjudication_backfills", [])
    records = list(records) if isinstance(records, list) else []
    records.append(dict(backfill_record))
    signal_metadata = []
    for signal in sorted(signal_manifest_paths):
        signal_manifest = _read_json_object(signal_manifest_paths[signal], label="signal manifest")
        value = signal_manifest.get("metadata")
        signal_metadata.append(dict(value) if isinstance(value, Mapping) else {})
    attempted = sum(int(value.get("planned_prompt_rows", 0)) for value in signal_metadata)
    rejected_rows = attempted - total_rows
    rejection_reasons: dict[str, int] = {}
    for value in signal_metadata:
        raw_reasons = value.get("rejection_reasons", {})
        if isinstance(raw_reasons, Mapping):
            for reason, count in raw_reasons.items():
                if isinstance(reason, str) and isinstance(count, int):
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count
    telemetry_items = [
        dict(value["llm_telemetry"])
        for value in signal_metadata
        if isinstance(value.get("llm_telemetry"), Mapping)
    ]
    metadata.update(
        {
            "accepted_rows": total_rows,
            "planned_prompt_rows": attempted,
            "rejected_rows": rejected_rows,
            "rejection_reasons": rejection_reasons,
            "response_quality": {
                "checked_rows": attempted,
                "accepted_rows": total_rows,
                "rejected_rows": rejected_rows,
                "rejection_reasons": rejection_reasons,
            },
            "llm_telemetry": aggregate_llm_telemetry(telemetry_items),
            "adjudication_backfills": records,
            **accepted_target_metadata(
                unit="rows",
                target_count=sum(entry["row_count"] for entry in manifest["datasets"]),
                accepted_count=total_rows,
                attempted_count=attempted,
                max_backfill_rounds=int(metadata.get("max_backfill_rounds", 0)),
                backfill_rounds=int(metadata.get("backfill_rounds", 0)),
            ),
        }
    )
    manifest["datasets"] = datasets
    manifest["total_rows"] = total_rows
    manifest["metadata"] = metadata
    for key in ("failure_status", "failure_reason", "run_failed"):
        manifest.pop(key, None)
        metadata.pop(key, None)
    _write_json_atomically(path, manifest)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_public_jsonl_atomically(path: Path, rows: list[Mapping[str, Any]]) -> None:
    validated = [validate_public_row(row) for row in rows]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in validated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
