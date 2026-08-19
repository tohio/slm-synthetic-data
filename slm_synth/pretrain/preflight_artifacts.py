#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from slm_synth.pretrain.artifacts.quality import artifact_fingerprint, artifact_structure_fingerprint, validate_artifact
from slm_synth.pretrain.grounded import FACTORY_MAP
from slm_synth.pretrain.generate import _rounded_batch_target_rows, _uncapped_grounded_target_rows
from slm_synth.paths import load_yaml_config, resolve_output_dir


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
            batch_size = int(mix_cfg.get("batch_size", cfg.get("generation", {}).get("batch_size", 32)))
            token_target, requested_rows, rounded_rows = _rounded_batch_target_rows(cfg, mix_cfg, batch_size)
            uncapped_requested_rows = _uncapped_grounded_target_rows(cfg, mix_cfg)
            factory = FACTORY_MAP[name]()
            families = Counter()
            exact_duplicates = 0
            quality_issues = []
            for index in range(rounded_rows):
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
                "requested_rows": uncapped_requested_rows,
                "planned_rows": requested_rows,
                "rounded_rows": rounded_rows,
                "max_unique_candidates": mix_cfg.get("max_unique_candidates"),
                "capacity_limited": requested_rows < uncapped_requested_rows,
                "exact_duplicates": exact_duplicates,
                "unique_structures": unique_structures,
                "family_counts": dict(sorted(families.items())),
                "quality_issue_count": len(quality_issues),
                "quality_issues": quality_issues,
            }
            reports.append(report)
            print(
                f"[preflight-artifacts] {name}: rows={rounded_rows}, exact_duplicates={exact_duplicates}, "
                f"structures={unique_structures}, quality_issues={len(quality_issues)}"
            )
    finally:
        connection.close()

    result = {"source_config": config, "signals": reports, "sqlite_index": str(db_path)}
    report_path = manifest_dir / "preflight_artifact_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[preflight-artifacts] Saved report: {report_path}")
    arithmetic_structure_reuse = any(
        row["signal"] == "arithmetic" and row["unique_structures"] != row["rounded_rows"]
        for row in reports
    )
    if arithmetic_structure_reuse:
        raise SystemExit(
            "Preflight failed: arithmetic candidate plan repeats structures before paid rendering."
        )
    if any(row["exact_duplicates"] or row["quality_issue_count"] for row in reports):
        raise SystemExit("Preflight failed: artifact duplicates or quality issues were found.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preflight every planned grounded artifact before paid rendering.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--signal", default=None)
    args = parser.parse_args()
    scan_plan(args.config, args.signal)
