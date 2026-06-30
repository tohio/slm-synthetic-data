#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_synth.pretrain.artifacts.base import GroundedArtifact
from slm_synth.pretrain.artifacts.quality import artifact_fingerprint, artifact_structure_fingerprint, validate_artifact
from slm_synth.paths import load_yaml_config, resolve_output_dir


def _iter_artifacts(batch_dir: Path):
    for path in sorted(batch_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("artifacts", []):
            yield GroundedArtifact(
                signal=row["signal"], family=row["family"], artifact_id=row["artifact_id"], payload=row["payload"]
            )


def scan_signal(output_dir: Path, signal: str) -> dict[str, Any]:
    batch_dir = output_dir / "manifests" / "grounded" / signal / "batches"
    artifacts = list(_iter_artifacts(batch_dir)) if batch_dir.exists() else []
    exact = Counter(artifact_fingerprint(artifact) for artifact in artifacts)
    structures = Counter(artifact_structure_fingerprint(artifact) for artifact in artifacts)
    families = Counter(artifact.family for artifact in artifacts)
    issues: list[dict[str, Any]] = []
    for artifact in artifacts:
        detected = validate_artifact(artifact)
        if detected:
            issues.append({"artifact_id": artifact.artifact_id, "issues": detected})
    total = len(artifacts)
    exact_duplicates = sum(count - 1 for count in exact.values() if count > 1)
    return {
        "signal": signal,
        "total_artifacts": total,
        "unique_artifacts": len(exact),
        "exact_duplicates": exact_duplicates,
        "exact_duplicate_rate": exact_duplicates / total if total else 0.0,
        "unique_structures": len(structures),
        "family_counts": dict(sorted(families.items())),
        "quality_issue_count": len(issues),
        "quality_issues": issues[:20],
    }


def main(config: str, signal: str | None = None) -> None:
    cfg = load_yaml_config(config)
    output_dir = resolve_output_dir(cfg)
    signals = [signal] if signal else list(cfg.get("mix", {}).keys())
    reports = [scan_signal(output_dir, current) for current in signals]
    for report in reports:
        pct = report["exact_duplicate_rate"] * 100.0
        print(
            f"[artifacts] {report['signal']}: total={report['total_artifacts']}, "
            f"unique={report['unique_artifacts']}, exact_duplicates={report['exact_duplicates']} ({pct:.2f}%), "
            f"structures={report['unique_structures']}, quality_issues={report['quality_issue_count']}"
        )
        print(f"[artifacts] {report['signal']} families={json.dumps(report['family_counts'], sort_keys=True)}")
    report_path = output_dir / "manifests" / "grounded" / "artifact_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"signals": reports}, indent=2), encoding="utf-8")
    print(f"[artifacts] Saved report: {report_path}")
    if any(report["quality_issue_count"] or report["exact_duplicates"] for report in reports):
        raise SystemExit("Grounded artifact report found quality issues or exact duplicate artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report grounded artifact diversity before/after rendering.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--signal", default=None)
    args = parser.parse_args()
    main(args.config, args.signal)
