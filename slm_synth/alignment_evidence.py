"""Shared run-manifest evidence collection for generic alignment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_deterministic_output_validation_summary(
    *,
    row_ids: set[str],
    manifest: dict[str, Any],
    run_manifest_path: Path | None,
) -> dict[str, Any]:
    """Collect accepted-row deterministic evidence from batch manifests."""
    evidence: dict[str, dict[str, Any]] = {}
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
            validation = (
                metadata.get("deterministic_output_validation")
                if isinstance(metadata, dict)
                else None
            )
            if not isinstance(validation, dict):
                evidence_errors.append(
                    f"batch manifest lacks deterministic output validation: {path}"
                )
                continue
            for row_id, result in validation.items():
                if row_id in evidence:
                    evidence_errors.append(
                        f"duplicate deterministic evidence for row: {row_id}"
                    )
                elif isinstance(row_id, str) and isinstance(result, dict):
                    evidence[row_id] = result
                else:
                    evidence_errors.append(
                        f"malformed deterministic evidence in: {path}"
                    )

    failed_ids = sorted(
        row_id
        for row_id in row_ids
        if row_id in evidence and evidence[row_id].get("status") != "passed"
    )
    missing_ids = sorted(row_ids - set(evidence))
    passed_ids = sorted(row_ids - set(failed_ids) - set(missing_ids))
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
        "public_row_count": len(row_ids),
        "passed_row_count": len(passed_ids),
        "failed_row_count": len(failed_ids),
        "missing_row_count": len(missing_ids),
        "passed_row_ids": passed_ids,
        "failed_row_ids": failed_ids,
        "missing_row_ids": missing_ids,
        "evidence_manifest_count": manifest_count,
        "evidence_errors": evidence_errors,
    }


def filter_validation_summary(
    summary: dict[str, Any], row_ids: set[str]
) -> dict[str, Any]:
    """Restrict one run-level evidence summary to a family or dimension."""
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


def deterministic_validation_blockers(
    summary: dict[str, Any], *, required: bool
) -> list[str]:
    """Return publication blockers for failed or absent deterministic evidence."""
    blockers: list[str] = []
    if summary["failed_row_count"]:
        blockers.append("deterministic_output_constraint_failed")
    if required and (
        summary["status"] in {"not_checked", "incomplete"}
        or summary["missing_row_count"]
        or summary["evidence_errors"]
    ):
        blockers.append("deterministic_output_validation_missing")
    return blockers


def build_quality_decision_summary(
    *,
    row_ids: set[str],
    manifest: dict[str, Any],
    run_manifest_path: Path | None,
) -> dict[str, Any]:
    """Collect final judge/reviewer decisions for every public alignment row."""
    evidence: dict[str, dict[str, Any]] = {}
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
            decisions = (
                metadata.get("quality_adjudication")
                if isinstance(metadata, dict)
                else None
            )
            if not isinstance(decisions, dict):
                evidence_errors.append(
                    f"batch manifest lacks quality adjudication: {path}"
                )
                continue
            for row_id, decision in decisions.items():
                if row_id in evidence:
                    evidence_errors.append(
                        f"duplicate adjudication evidence for row: {row_id}"
                    )
                elif isinstance(row_id, str) and isinstance(decision, dict):
                    evidence[row_id] = decision
                else:
                    evidence_errors.append(f"malformed adjudication evidence in: {path}")

    failed_ids = sorted(
        row_id
        for row_id in row_ids
        if row_id in evidence and not _quality_decision_passes(evidence[row_id])
    )
    missing_ids = sorted(row_ids - set(evidence))
    passed_ids = sorted(row_ids - set(failed_ids) - set(missing_ids))
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
        "public_row_count": len(row_ids),
        "passed_row_count": len(passed_ids),
        "failed_row_count": len(failed_ids),
        "missing_row_count": len(missing_ids),
        "passed_row_ids": passed_ids,
        "failed_row_ids": failed_ids,
        "missing_row_ids": missing_ids,
        "evidence_manifest_count": manifest_count,
        "evidence_errors": evidence_errors,
    }


def quality_decision_blockers(
    summary: dict[str, Any], *, required: bool
) -> list[str]:
    """Return publication blockers for failed or absent final quality evidence."""
    blockers: list[str] = []
    if summary["failed_row_count"]:
        blockers.append("semantic_adjudication_failed")
    if required and (
        summary["status"] in {"not_checked", "incomplete"}
        or summary["missing_row_count"]
        or summary["evidence_errors"]
    ):
        blockers.append("semantic_adjudication_missing")
    return blockers


def _quality_decision_passes(decision: dict[str, Any]) -> bool:
    return (
        decision.get("accepted") is True
        and decision.get("assessable") is True
        and decision.get("judge_accepted") is True
        and decision.get("reviewed") is True
        and decision.get("reviewer_agreed") is True
    )


def _resolve_manifest_path(raw_path: str, run_manifest_path: Path | None) -> Path | None:
    path = Path(raw_path)
    if path.is_file():
        return path
    if run_manifest_path is not None:
        candidate = run_manifest_path.parent / path
        if candidate.is_file():
            return candidate
    return None
