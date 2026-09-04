#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from slm_synth.pretrain.artifacts.quality import artifact_fingerprint, artifact_structure_fingerprint, validate_artifact
from slm_synth.pretrain.artifacts.planning import (
    ScalableGroundedArtifactFactory,
    build_artifact_factory,
    configured_candidate_capacity,
)
from slm_synth.pretrain.generate import _planned_grounded_target_rows
from slm_synth.paths import load_yaml_config, resolve_output_dir


def _sampled_plan_indices(
    *,
    factory: object,
    planned_rows: int,
    max_rows: int,
) -> tuple[list[int], str]:
    """Return a deterministic full or sampled preflight plan.

    Finite plans remain fully enumerated. Large scalable plans validate every
    finite capability anchor plus an evenly distributed sample of derived
    profiles, while capacity coverage is checked analytically.
    """

    if planned_rows < 0:
        raise ValueError("planned_rows must be non-negative")
    if max_rows < 1:
        raise ValueError("preflight_max_artifacts must be positive")
    if planned_rows <= max_rows or not isinstance(
        factory, ScalableGroundedArtifactFactory
    ):
        return list(range(planned_rows)), "full"

    anchor_rows = min(factory.base_capacity, planned_rows)
    indices = list(range(anchor_rows))
    derived_budget = max_rows - anchor_rows
    if derived_budget < 1:
        raise ValueError(
            "preflight_max_artifacts must exceed the finite capability-anchor "
            f"count ({anchor_rows}) for scalable plans"
        )

    derived_start = anchor_rows
    derived_count = planned_rows - derived_start
    if derived_count <= derived_budget:
        indices.extend(range(derived_start, planned_rows))
        return indices, "full"

    if derived_budget == 1:
        indices.append(derived_start)
    else:
        span = derived_count - 1
        indices.extend(
            derived_start + (slot * span) // (derived_budget - 1)
            for slot in range(derived_budget)
        )
    return indices, "sampled"


def scan_plan(config: str, signal: str | None = None) -> dict:
    cfg = load_yaml_config(config)
    out = resolve_output_dir(cfg)
    manifest_dir = out / "manifests" / "grounded"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    db_path = manifest_dir / "preflight_artifacts.sqlite"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE exact (signal TEXT, digest TEXT, artifact_id TEXT, PRIMARY KEY(signal, digest))")
    connection.execute("CREATE TABLE structures (signal TEXT, family TEXT, digest TEXT, count INTEGER, PRIMARY KEY(signal, digest))")

    signal_names = [signal] if signal else list(cfg.get("mix", {}))
    reports = []
    try:
        for name in signal_names:
            mix_cfg = cfg["mix"][name]
            if mix_cfg.get("architecture") != "grounded":
                continue
            factory = build_artifact_factory(name, mix_cfg)
            capacity = configured_candidate_capacity(
                name,
                mix_cfg,
                factory=factory,
            )
            token_target, requested_rows, planned_rows = _planned_grounded_target_rows(
                cfg,
                mix_cfg,
                candidate_capacity=capacity,
            )
            avg_tokens_per_sample = float(
                mix_cfg.get("avg_tokens_per_sample", cfg.get("generation", {}).get("avg_tokens_per_sample", 100))
            )
            estimated_capacity_tokens = capacity * avg_tokens_per_sample
            generation_cfg = cfg.get("generation", {})
            if not isinstance(generation_cfg, dict):
                generation_cfg = {}
            preflight_max_artifacts = int(
                generation_cfg.get(
                    "preflight_max_artifacts",
                    cfg.get("preflight_max_artifacts", 10_000),
                )
            )
            preflight_indices, preflight_mode = _sampled_plan_indices(
                factory=factory,
                planned_rows=planned_rows,
                max_rows=preflight_max_artifacts,
            )
            preflight_rows = len(preflight_indices)
            families = Counter()
            exact_duplicates = 0
            quality_issues = []
            for index in preflight_indices:
                artifact = factory.build(index)
                families[artifact.family] += 1
                issues = validate_artifact(artifact)
                if issues and len(quality_issues) < 20:
                    quality_issues.append({"artifact_id": artifact.artifact_id, "issues": issues})
                digest = artifact_fingerprint(artifact)
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO exact(signal, digest, artifact_id) VALUES (?, ?, ?)",
                    (name, digest, artifact.artifact_id),
                )
                if cursor.rowcount == 0:
                    exact_duplicates += 1
                structure = artifact_structure_fingerprint(artifact)
                connection.execute(
                    "INSERT INTO structures(signal, family, digest, count) VALUES (?, ?, ?, 1) "
                    "ON CONFLICT(signal, digest) DO UPDATE SET count = count + 1",
                    (name, artifact.family, structure),
                )
            connection.commit()
            unique_structures = connection.execute("SELECT COUNT(*) FROM structures WHERE signal = ?", (name,)).fetchone()[0]
            report = {
                "signal": name,
                "target_tokens_estimate": token_target,
                "requested_rows": requested_rows,
                "planned_rows": planned_rows,
                "preflight_rows": preflight_rows,
                "preflight_mode": preflight_mode,
                "preflight_max_artifacts": preflight_max_artifacts,
                "candidate_planner": str(mix_cfg.get("candidate_planner", "finite")),
                "base_candidate_capacity": int(
                    getattr(factory, "base_capacity", capacity)
                ),
                "derivation_profile_count": int(
                    getattr(factory, "profile_count", 0)
                ),
                "max_unique_candidates": capacity,
                "estimated_capacity_tokens": estimated_capacity_tokens,
                "capacity_covers_target": estimated_capacity_tokens is None or estimated_capacity_tokens >= token_target,
                "capacity_limited": planned_rows < requested_rows,
                "exact_duplicates": exact_duplicates,
                "unique_structures": unique_structures,
                "family_counts": dict(sorted(families.items())),
                "quality_issue_count": len(quality_issues),
                "quality_issues": quality_issues,
            }
            reports.append(report)
            print(
                f"[preflight-artifacts] {name}: planned_rows={planned_rows}, "
                f"preflight_rows={preflight_rows}, mode={preflight_mode}, "
                f"exact_duplicates={exact_duplicates}, "
                f"structures={unique_structures}, quality_issues={len(quality_issues)}"
            )
    finally:
        connection.close()

    result = {"source_config": config, "signals": reports, "sqlite_index": str(db_path)}
    report_path = manifest_dir / "preflight_artifact_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[preflight-artifacts] Saved report: {report_path}")
    repaired_signal_structure_reuse = any(
        row["signal"] in {
            "arithmetic",
            "task_code",
            "educational_qa_mcq_math",
            "educational_qa_mcq_general",
            "factual_restraint",
        }
        and row["unique_structures"] != row["preflight_rows"]
        for row in reports
    )
    if repaired_signal_structure_reuse:
        raise SystemExit(
            "Preflight failed: a quality-capped candidate plan repeats structures before paid rendering."
        )
    if any(row["exact_duplicates"] or row["quality_issue_count"] for row in reports):
        raise SystemExit("Preflight failed: artifact duplicates or quality issues were found.")
    capacity_shortfalls = [
        row["signal"] for row in reports if not row["capacity_covers_target"]
    ]
    if capacity_shortfalls:
        raise SystemExit(
            "Preflight failed: candidate inventory cannot cover the accepted-token target "
            f"for signals={capacity_shortfalls}."
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preflight every planned grounded artifact before paid rendering.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--signal", default=None)
    args = parser.parse_args()
    scan_plan(args.config, args.signal)
