"""Coverage reporting helpers for synthetic SFT datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_synth.alignment_tokens import estimate_sft_tokens
from slm_synth.sft.acceptance import build_sft_content_summary, partition_unique_sft_rows
from slm_synth.sft.publication_quality import build_publication_quality_summary
from slm_synth.sft.schema import validate_sft_row
from slm_synth.taxonomy.holdouts import HoldoutRegistry


def build_coverage_report(
    paths: list[str | Path],
    *,
    holdout_registry: HoldoutRegistry | None = None,
    run_manifest: str | Path | None = None,
    require_holdout_check: bool = True,
    require_semantic_adjudication: bool | None = None,
) -> dict[str, Any]:
    """Build aggregate and per-family SFT acceptance and coverage reporting."""
    dataset_paths = _resolve_jsonl_paths(paths)
    rows: list[dict[str, Any]] = []
    file_counts: dict[str, int] = {}
    validation_errors: list[dict[str, Any]] = []

    for path in dataset_paths:
        file_rows, row_count, errors = _read_jsonl_with_diagnostics(path)
        file_counts[str(path)] = row_count
        rows.extend(file_rows)
        validation_errors.extend(errors)

    content = build_sft_content_summary(rows)
    publication_quality = build_publication_quality_summary(rows)
    unique_rows, visible_acceptance = partition_unique_sft_rows(rows)
    manifest = _read_run_manifest(run_manifest)
    manifest_metadata = manifest.get("metadata", {}) if manifest else {}
    attempted_rows = _non_negative_int(
        manifest_metadata.get("attempted_rows"),
        len(rows),
    )
    duplicate_rows = max(
        _non_negative_int(manifest_metadata.get("duplicate_rows"), 0),
        visible_acceptance["duplicate_rows"],
    )
    rejected_rows = _non_negative_int(manifest_metadata.get("rejected_rows"), 0)
    accepted_rows = len(unique_rows)
    estimated_tokens = sum(estimate_sft_tokens(row) for row in unique_rows)
    candidate_rows = _non_negative_int(manifest_metadata.get("candidate_rows"), attempted_rows)

    holdouts = _build_holdout_summary(rows, holdout_registry)
    semantic_adjudication = _build_semantic_adjudication_summary(
        rows=rows,
        manifest=manifest,
        run_manifest_path=Path(run_manifest) if run_manifest is not None else None,
    )
    semantic_required = (
        run_manifest is not None
        if require_semantic_adjudication is None
        else require_semantic_adjudication
    )
    validation = _build_validation_summary(validation_errors)
    blockers = _publish_blockers(
        content=content,
        publication_quality=publication_quality,
        validation=validation,
        semantic_adjudication=semantic_adjudication,
        holdouts=holdouts,
        manifest_metadata=manifest_metadata,
        require_holdout_check=require_holdout_check,
        require_semantic_adjudication=semantic_required,
    )
    if not unique_rows:
        blockers.append("empty_dataset")

    return {
        "dataset_type": "sft",
        "row_count": sum(file_counts.values()),
        "valid_row_count": len(rows),
        "files": file_counts,
        "task_families": _count_metadata(rows, "task_family"),
        "interaction_modes": _count_list_metadata(rows, "interaction_modes"),
        "output_modes": _count_metadata(rows, "output_mode"),
        "context_modes": _count_metadata(rows, "context_mode"),
        "template_families": _count_metadata(rows, "template_family"),
        "difficulty_counts": _count_metadata(rows, "difficulty", stringify_keys=True),
        "content_uniqueness": content,
        **publication_quality,
        "validation": validation,
        "semantic_adjudication": semantic_adjudication,
        "holdouts": holdouts,
        "acceptance": {
            "attempted_rows": attempted_rows,
            "candidate_rows": candidate_rows,
            "accepted_rows": accepted_rows,
            "estimated_tokens": estimated_tokens,
            "rejected_rows": rejected_rows,
            "rejection_reason_counts": _count_mapping(
                manifest_metadata.get("rejection_reason_counts")
            ),
            "rejection_diagnostics": _rejection_diagnostics(manifest_metadata),
            "duplicate_rows": duplicate_rows,
            "publish_ready": not blockers,
            "publish_blockers": blockers,
        },
        "families": _build_family_reports(
            rows,
            manifest_metadata=manifest_metadata,
            holdout_registry=holdout_registry,
            require_holdout_check=require_holdout_check,
            semantic_adjudication=semantic_adjudication,
            require_semantic_adjudication=semantic_required,
        ),
    }


def require_publish_ready_report(report: dict[str, Any], *, artifact_name: str = "SFT") -> None:
    """Reject an SFT artifact whose acceptance report contains publish blockers."""
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{artifact_name} coverage report is missing acceptance reporting")
    blockers = acceptance.get("publish_blockers")
    if acceptance.get("publish_ready") is not True or not isinstance(blockers, list) or blockers:
        detail = ", ".join(str(blocker) for blocker in blockers) if isinstance(blockers, list) else "unknown"
        raise ValueError(f"{artifact_name} acceptance report is not publish-ready: {detail}")


def write_coverage_report(*, report: dict[str, Any], path: str | Path) -> Path:
    """Write a coverage report JSON file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _resolve_jsonl_paths(paths: list[str | Path]) -> list[Path]:
    if not paths:
        raise ValueError("at least one input path is required")

    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.extend(sorted(candidate for candidate in path.glob("*.jsonl") if candidate.is_file()))
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"input path does not exist: {path}")

    if not resolved:
        raise ValueError("no JSONL dataset files found")
    return resolved


def _count_metadata(
    rows: list[dict[str, Any]],
    field: str,
    *,
    stringify_keys: bool = False,
) -> dict[str, int]:
    counter = Counter(row["metadata"][field] for row in rows)
    if stringify_keys:
        return {str(key): counter[key] for key in sorted(counter)}
    return {key: counter[key] for key in sorted(counter)}


def _count_list_metadata(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(item for row in rows for item in row["metadata"][field])
    return {key: counter[key] for key in sorted(counter)}


def _build_family_reports(
    rows: list[dict[str, Any]],
    *,
    manifest_metadata: dict[str, Any],
    holdout_registry: HoldoutRegistry | None,
    require_holdout_check: bool,
    semantic_adjudication: dict[str, Any],
    require_semantic_adjudication: bool,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    configured = manifest_metadata.get("candidate_rows_per_family", {})
    family_names = {row["metadata"]["task_family"] for row in rows}
    if isinstance(configured, dict):
        family_names.update(configured)
    for family in sorted(family_names):
        family_rows = [row for row in rows if row["metadata"]["task_family"] == family]
        content = build_sft_content_summary(family_rows)
        publication_quality = build_publication_quality_summary(family_rows)
        unique_rows, visible_acceptance = partition_unique_sft_rows(family_rows)
        attempted = _family_count(manifest_metadata, "attempted_rows_per_family", family, len(family_rows))
        duplicates = max(
            _family_count(manifest_metadata, "duplicate_rows_per_family", family, 0),
            visible_acceptance["duplicate_rows"],
        )
        rejected = _family_count(
            manifest_metadata,
            "rejected_rows_per_family",
            family,
            max(attempted - len(unique_rows) - duplicates, 0),
        )
        candidate_rows = _family_count(
            manifest_metadata,
            "candidate_rows_per_family",
            family,
            attempted,
        )
        holdouts = _build_holdout_summary(family_rows, holdout_registry)
        blockers = _publish_blockers(
            content=content,
            publication_quality=publication_quality,
            validation={"invalid_row_count": 0, "invalid_tool_or_role_sequence_count": 0},
            semantic_adjudication=_filter_semantic_adjudication(
                semantic_adjudication, {row["id"] for row in family_rows}
            ),
            holdouts=holdouts,
            manifest_metadata={},
            require_holdout_check=require_holdout_check,
            require_semantic_adjudication=require_semantic_adjudication,
        )
        if not unique_rows:
            blockers.append("empty_family")
        reports[family] = {
            "row_count": len(family_rows),
            "interaction_modes": _count_list_metadata(family_rows, "interaction_modes"),
            "output_modes": _count_metadata(family_rows, "output_mode"),
            "context_modes": _count_metadata(family_rows, "context_mode"),
            "template_families": _count_metadata(family_rows, "template_family"),
            "difficulty_counts": _count_metadata(family_rows, "difficulty", stringify_keys=True),
            "content_uniqueness": content,
            **publication_quality,
            "holdouts": holdouts,
            "acceptance": {
                "attempted_rows": attempted,
                "candidate_rows": candidate_rows,
                "accepted_rows": len(unique_rows),
                "estimated_tokens": sum(estimate_sft_tokens(row) for row in unique_rows),
                "rejected_rows": rejected,
                "rejection_diagnostics": _rejection_diagnostics(
                    manifest_metadata, family=family
                ),
                "duplicate_rows": duplicates,
                "publish_ready": not blockers,
                "publish_blockers": blockers,
            },
        }
    return reports


def _build_holdout_summary(
    rows: list[dict[str, Any]],
    registry: HoldoutRegistry | None,
) -> dict[str, Any]:
    if registry is None:
        return {"status": "not_checked", "collision_count": None, "collision_ids": []}

    collision_ids: list[str] = []
    for row in rows:
        if any(
            message["role"] == "user" and registry.contains_prompt(message["content"])
            for message in row["messages"]
        ):
            collision_ids.append(row["id"])
    return {
        "status": "checked",
        "collision_count": len(collision_ids),
        "collision_ids": collision_ids,
    }


def _publish_blockers(
    *,
    content: dict[str, Any],
    publication_quality: dict[str, Any],
    validation: dict[str, Any],
    semantic_adjudication: dict[str, Any],
    holdouts: dict[str, Any],
    manifest_metadata: dict[str, Any],
    require_holdout_check: bool,
    require_semantic_adjudication: bool,
) -> list[str]:
    blockers: list[str] = []
    for key, blocker in (
        ("ids", "duplicate_ids"),
        ("prompts", "duplicate_prompts"),
        ("conversations", "duplicate_conversations"),
    ):
        if content[key]["duplicate_count"]:
            blockers.append(blocker)
    near_duplicates = publication_quality["near_duplicates"]
    if near_duplicates["prompts"]["pair_count"]:
        blockers.append("near_duplicate_prompts")
    if near_duplicates["conversations"]["pair_count"]:
        blockers.append("near_duplicate_conversations")
    if publication_quality["assistant_response_clusters"]["cluster_count"]:
        blockers.append("repeated_assistant_response_clusters")
    if publication_quality["template_concentration"]["concentrated_template_count"]:
        blockers.append("template_concentration")
    if validation["invalid_row_count"]:
        blockers.append("invalid_public_rows")
    if validation["invalid_tool_or_role_sequence_count"]:
        blockers.append("invalid_tool_or_role_sequences")
    if semantic_adjudication["failed_row_count"]:
        blockers.append("semantic_adjudication_failed")
    if require_semantic_adjudication and (
        semantic_adjudication["status"] in {"not_checked", "incomplete"}
        or semantic_adjudication["missing_row_count"]
        or semantic_adjudication["evidence_errors"]
    ):
        blockers.append("semantic_adjudication_missing")
    if holdouts["status"] == "checked" and holdouts["collision_count"]:
        blockers.append("holdout_collisions")
    if require_holdout_check and holdouts["status"] != "checked":
        blockers.append("holdouts_not_checked")
    if manifest_metadata and manifest_metadata.get("publish_ready") is False:
        blockers.append("run_manifest_not_publish_ready")
    return blockers


def _rejection_diagnostics(
    metadata: dict[str, Any], *, family: str | None = None
) -> list[dict[str, Any]]:
    value = metadata.get("rejection_diagnostics", [])
    if not isinstance(value, list):
        return []
    diagnostics = [dict(item) for item in value if isinstance(item, dict)]
    if family is not None:
        diagnostics = [item for item in diagnostics if item.get("family") == family]
    return diagnostics


def _read_jsonl_with_diagnostics(
    path: Path,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    row_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row_count += 1
        row_id: str | None = None
        try:
            value = json.loads(line)
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                row_id = value["id"]
            rows.append(validate_sft_row(value))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(
                {
                    "path": str(path),
                    "line": line_number,
                    "id": row_id,
                    "error": str(exc),
                    "tool_or_role_sequence": _is_tool_or_role_error(str(exc)),
                }
            )
    return rows, row_count, errors


def _is_tool_or_role_error(message: str) -> bool:
    normalized = message.casefold()
    return any(
        marker in normalized
        for marker in (
            "role",
            "message sequence",
            "messages",
            "tool",
            "call id",
            "tool_call",
            "interaction_modes",
        )
    )


def _build_validation_summary(errors: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "clean" if not errors else "failed",
        "invalid_row_count": len(errors),
        "invalid_tool_or_role_sequence_count": sum(
            error["tool_or_role_sequence"] is True for error in errors
        ),
        "errors": errors[:20],
    }


def _read_run_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SFT run manifest must contain a JSON object: {manifest_path}")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"SFT run manifest metadata must contain an object: {manifest_path}")
    return value


def _build_semantic_adjudication_summary(
    *, rows: list[dict[str, Any]], manifest: dict[str, Any], run_manifest_path: Path | None
) -> dict[str, Any]:
    public_ids = {row["id"] for row in rows}
    decisions: dict[str, dict[str, Any]] = {}
    evidence_errors: list[str] = []
    manifest_count = 0
    datasets = manifest.get("datasets") if manifest else None
    if isinstance(datasets, list):
        raw_paths = [
            raw_path
            for dataset in datasets
            if isinstance(dataset, dict)
            for raw_path in dataset.get("batch_manifests", [])
            if isinstance(raw_path, str)
        ]
        for raw_path in raw_paths:
            path = _resolve_manifest_path(raw_path, run_manifest_path)
            if path is None:
                evidence_errors.append(f"missing batch manifest: {raw_path}")
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                evidence_errors.append(f"unreadable batch manifest {path}: {exc}")
                continue
            manifest_count += 1
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            quality = metadata.get("quality_adjudication") if isinstance(metadata, dict) else None
            if not isinstance(quality, dict):
                evidence_errors.append(f"batch manifest lacks quality adjudication: {path}")
                continue
            for row_id, decision in quality.items():
                if row_id in decisions:
                    evidence_errors.append(f"duplicate adjudication evidence for row: {row_id}")
                elif isinstance(row_id, str) and isinstance(decision, dict):
                    decisions[row_id] = decision
                else:
                    evidence_errors.append(f"malformed adjudication evidence in: {path}")

    failed_ids = sorted(
        row_id for row_id in public_ids if row_id in decisions and not _decision_passes(decisions[row_id])
    )
    missing_ids = sorted(public_ids - set(decisions))
    passed_ids = sorted(public_ids - set(failed_ids) - set(missing_ids))
    if not manifest:
        status = "not_checked"
    elif missing_ids or evidence_errors:
        status = "incomplete"
    elif failed_ids:
        status = "failed"
    else:
        status = "checked"
    return {
        "status": status,
        "public_row_count": len(public_ids),
        "passed_row_count": len(passed_ids),
        "failed_row_count": len(failed_ids),
        "missing_row_count": len(missing_ids),
        "passed_row_ids": passed_ids,
        "failed_row_ids": failed_ids,
        "missing_row_ids": missing_ids,
        "evidence_manifest_count": manifest_count,
        "evidence_errors": evidence_errors,
    }


def _resolve_manifest_path(raw_path: str, run_manifest_path: Path | None) -> Path | None:
    path = Path(raw_path)
    if path.is_file():
        return path
    if run_manifest_path is not None:
        candidate = run_manifest_path.parent / path
        if candidate.is_file():
            return candidate
    return None


def _decision_passes(decision: dict[str, Any]) -> bool:
    scores = decision.get("scores")
    constraints = decision.get("constraint_results")
    return (
        decision.get("accepted") is True
        and isinstance(scores, dict)
        and bool(scores)
        and all(
            isinstance(score, int) and not isinstance(score, bool) and score >= 3
            for score in scores.values()
        )
        and isinstance(constraints, list)
        and all(isinstance(item, dict) and item.get("passed") is True for item in constraints)
    )


def _filter_semantic_adjudication(
    summary: dict[str, Any], row_ids: set[str]
) -> dict[str, Any]:
    passed = sorted(row_ids & set(summary["passed_row_ids"]))
    failed = sorted(row_ids & set(summary["failed_row_ids"]))
    missing = sorted(row_ids & set(summary["missing_row_ids"]))
    if summary["status"] == "not_checked":
        status = "not_checked"
    elif missing or summary["evidence_errors"]:
        status = "incomplete"
    elif failed:
        status = "failed"
    else:
        status = "checked"
    return {
        **summary,
        "status": status,
        "public_row_count": len(row_ids),
        "passed_row_count": len(passed),
        "failed_row_count": len(failed),
        "missing_row_count": len(missing),
        "passed_row_ids": passed,
        "failed_row_ids": failed,
        "missing_row_ids": missing,
    }


def _non_negative_int(*values: Any) -> int:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _family_count(metadata: dict[str, Any], field: str, family: str, fallback: int) -> int:
    values = metadata.get(field)
    if isinstance(values, dict):
        value = values.get(family)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return fallback


def _count_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: count
        for key, count in sorted(value.items())
        if isinstance(key, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
    }
