"""Apply explicit human decisions to repeated Distillation-SFT responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from slm_synth.distillation_sft.response_diversity import (
    build_response_diversity_summary,
    response_cluster_member_fingerprint,
)
from slm_synth.distillation_sft.response_quality import is_response_machine_verified
from slm_synth.distillation_sft.schema import validate_public_row

ADJUDICATION_SCHEMA_VERSION = 1
ADJUDICATION_DECISIONS = frozenset({"keep", "reject"})


def apply_response_cluster_adjudications(
    *,
    dataset_dir: str | Path,
    adjudications_path: str | Path,
    rejected_dir: str | Path,
    run_manifest_path: str | Path,
) -> dict[str, Any]:
    """Apply complete member-level decisions and quarantine rejected rows.

    Validation completes before any file is changed. Decisions are bound to a
    fingerprint of the signal and full public row, so a row ID alone cannot
    authorize a keep or rejection.
    """
    dataset_root = Path(dataset_dir)
    files = sorted(path for path in dataset_root.glob("*.jsonl") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No distillation JSONL files found in {dataset_root}")

    rows_by_file, rows_by_member = _load_rows(files)
    required_members = _required_review_members(files, rows_by_member)
    decisions = _load_adjudications(adjudications_path)
    _require_exact_decision_coverage(required_members, decisions)

    rejected_records: list[dict[str, Any]] = []
    kept_rows_by_file: dict[Path, list[dict[str, Any]]] = {}
    kept_count = 0
    rejected_count = 0
    for file_path, rows in rows_by_file.items():
        signal = file_path.stem.split(".batch", 1)[0]
        kept_rows: list[dict[str, Any]] = []
        for row in rows:
            member_fingerprint = response_cluster_member_fingerprint(signal=signal, row=row)
            decision = decisions.get(member_fingerprint)
            if decision is None or decision["decision"] == "keep":
                kept_rows.append(row)
                kept_count += 1
                continue
            rejected_count += 1
            rejected_records.append(
                {
                    "rejection_reason": "repeated_response_cluster_adjudication",
                    "response_fingerprint": required_members[member_fingerprint]["response_fingerprint"],
                    "member_fingerprint": member_fingerprint,
                    "decision_reason": decision["reason"],
                    "signal": signal,
                    "row": row,
                }
            )
        kept_rows_by_file[file_path] = kept_rows

    run_manifest_path = Path(run_manifest_path)
    updated_run_manifest = _build_updated_run_manifest(
        path=run_manifest_path,
        kept_rows_by_file=kept_rows_by_file,
        rejected_count=rejected_count,
    )

    for file_path, rows in kept_rows_by_file.items():
        _write_jsonl_atomically(file_path, rows)

    rejected_path = Path(rejected_dir) / "repeated_response_adjudications.jsonl"
    _write_unvalidated_jsonl_atomically(rejected_path, rejected_records)
    _write_json_atomically(run_manifest_path, updated_run_manifest)
    return {
        "reviewed_rows": len(required_members),
        "kept_rows": kept_count,
        "rejected_rows": rejected_count,
        "rejected_path": str(rejected_path),
    }


def _build_updated_run_manifest(
    *,
    path: Path,
    kept_rows_by_file: Mapping[Path, list[dict[str, Any]]],
    rejected_count: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError(f"distillation run manifest is invalid: {path}")
    counts_by_signal = {
        file_path.stem.split(".batch", 1)[0]: len(rows)
        for file_path, rows in kept_rows_by_file.items()
    }
    manifest_signals = {item.get("signal") for item in payload["datasets"] if isinstance(item, dict)}
    if manifest_signals != set(counts_by_signal):
        raise ValueError("adjudicated dataset signals do not match the run manifest")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"distillation run manifest metadata is invalid: {path}")
    prior_accepted = metadata.get("accepted_rows")
    if not isinstance(prior_accepted, int) or prior_accepted < 0:
        raise ValueError(f"distillation run manifest accepted_rows is invalid: {path}")
    metadata.setdefault("generated_accepted_rows", prior_accepted)
    metadata["curation_rejected_rows"] = metadata.get("curation_rejected_rows", 0) + rejected_count
    metadata["accepted_rows"] = sum(counts_by_signal.values())
    metadata["curated_rows_per_signal"] = dict(sorted(counts_by_signal.items()))
    for item in payload["datasets"]:
        item["row_count"] = counts_by_signal[item["signal"]]
    payload["total_rows"] = metadata["accepted_rows"]
    return payload


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _load_rows(
    files: list[Path],
) -> tuple[dict[Path, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    rows_by_file: dict[Path, list[dict[str, Any]]] = {}
    rows_by_member: dict[str, dict[str, Any]] = {}
    for file_path in files:
        signal = file_path.stem.split(".batch", 1)[0]
        rows: list[dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL in {file_path} at line {line_number}: {exc}"
                    ) from exc
                row = validate_public_row(value)
                fingerprint = response_cluster_member_fingerprint(signal=signal, row=row)
                if fingerprint in rows_by_member:
                    raise ValueError(
                        "duplicate content-bound response-cluster member fingerprint: "
                        f"{fingerprint}"
                    )
                rows.append(row)
                rows_by_member[fingerprint] = {"signal": signal, "row": row}
        rows_by_file[file_path] = rows
    return rows_by_file, rows_by_member


def _required_review_members(
    files: list[Path],
    rows_by_member: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    summary = build_response_diversity_summary(files)
    required: dict[str, dict[str, str]] = {}
    for cluster in summary["repeated_response_clusters"]:
        members = cluster["members"]
        machine_verified = all(
            is_response_machine_verified(
                signal=rows_by_member[member["member_fingerprint"]]["signal"],
                row=rows_by_member[member["member_fingerprint"]]["row"],
            )
            for member in members
        )
        if machine_verified:
            continue
        for member in members:
            required[member["member_fingerprint"]] = {
                "response_fingerprint": cluster["response_fingerprint"],
                "row_id": member["id"],
                "signal": member["signal"],
            }
    return required


def _load_adjudications(path: str | Path) -> dict[str, dict[str, str]]:
    adjudications_path = Path(path)
    try:
        payload = json.loads(adjudications_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid adjudication JSON in {adjudications_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("adjudication file must contain a JSON object")
    if payload.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
        raise ValueError(f"adjudication schema_version must be {ADJUDICATION_SCHEMA_VERSION}")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("adjudication decisions must be a list")

    decisions: dict[str, dict[str, str]] = {}
    for index, value in enumerate(raw_decisions):
        if not isinstance(value, Mapping):
            raise ValueError(f"adjudication decision {index} must be an object")
        fingerprint = value.get("member_fingerprint")
        decision = value.get("decision")
        reason = value.get("reason")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"adjudication decision {index} has an invalid member_fingerprint")
        if decision not in ADJUDICATION_DECISIONS:
            raise ValueError(f"adjudication decision {index} must be 'keep' or 'reject'")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"adjudication decision {index} requires a non-empty reason")
        if fingerprint in decisions:
            raise ValueError(f"duplicate adjudication for member_fingerprint {fingerprint}")
        decisions[fingerprint] = {"decision": decision, "reason": reason.strip()}
    return decisions


def _require_exact_decision_coverage(
    required: Mapping[str, Mapping[str, str]],
    decisions: Mapping[str, Mapping[str, str]],
) -> None:
    missing = sorted(set(required) - set(decisions))
    extra = sorted(set(decisions) - set(required))
    if not missing and not extra:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing={len(missing)} ({', '.join(value[:12] for value in missing[:3])})")
    if extra:
        details.append(f"unknown_or_stale={len(extra)} ({', '.join(value[:12] for value in extra[:3])})")
    raise ValueError("response-cluster adjudications are incomplete or stale: " + "; ".join(details))


def _write_jsonl_atomically(path: Path, rows: list[Mapping[str, Any]]) -> None:
    validated_rows = [validate_public_row(row) for row in rows]
    _write_unvalidated_jsonl_atomically(path, validated_rows)


def _write_unvalidated_jsonl_atomically(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)
