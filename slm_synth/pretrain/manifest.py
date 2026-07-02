"""Run-level manifest helpers for synthetic pretraining data."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from slm_synth.pretrain.record_quality import SIGNAL_FROM_FILE

PRETRAIN_STAGES = ("raw", "validated", "deduped", "rejected")


def build_run_manifest(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    generation_run: str | None = None,
    stages: tuple[str, ...] = PRETRAIN_STAGES,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only manifest summarizing pretrain stage outputs."""
    cfg = _load_yaml_config(config_path)
    resolved_output_dir = Path(output_dir) if output_dir is not None else _resolve_output_dir(cfg)
    run_name = _resolve_generation_run(cfg=cfg, output_dir=resolved_output_dir, generation_run=generation_run)
    stage_payloads = {
        stage: _summarize_stage(resolved_output_dir / stage)
        for stage in stages
    }

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "pretrain",
        "generation_run": run_name,
        "config_path": str(Path(config_path)),
        "output_dir": str(resolved_output_dir),
        "stages": stage_payloads,
        "signals": _summarize_signals(stage_payloads),
        "metadata": dict(metadata or {}),
    }


def write_run_manifest(
    *,
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Write a pretrain run manifest JSON file and return its path."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def default_manifest_path(
    *,
    output_dir: str | Path,
    generation_run: str,
) -> Path:
    """Return the default pretrain run-level manifest path."""
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    return Path(output_dir) / "manifests" / f"{generation_run.strip()}.manifest.json"


def generate_run_manifest(
    *,
    config_path: str | Path,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    generation_run: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Build and write a pretrain run manifest from a config and existing outputs."""
    manifest = build_run_manifest(
        config_path=config_path,
        output_dir=output_dir,
        generation_run=generation_run,
        metadata=metadata,
    )
    manifest_path = Path(output) if output is not None else default_manifest_path(
        output_dir=manifest["output_dir"],
        generation_run=manifest["generation_run"],
    )
    return write_run_manifest(manifest_path=manifest_path, manifest=manifest)


def _resolve_generation_run(
    *,
    cfg: Mapping[str, Any],
    output_dir: Path,
    generation_run: str | None,
) -> str:
    if generation_run is not None and generation_run.strip():
        return generation_run.strip()
    configured = cfg.get("run_name") or cfg.get("run_id")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if output_dir.name:
        return output_dir.name
    raise ValueError("could not resolve generation_run from config or output_dir")


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("pretrain config must contain a YAML mapping")
    return value


def _resolve_output_dir(cfg: Mapping[str, Any]) -> Path:
    raw = str(cfg.get("output_dir", "")).strip()
    if not raw:
        raise ValueError("pretrain config is missing required key: output_dir")

    os.environ.setdefault("DATA_DIR", "data/runs")
    return Path(os.path.expandvars(raw))


def _summarize_stage(stage_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    total_rows = 0
    jsonl_files = sorted(stage_dir.glob("*.jsonl")) if stage_dir.exists() else []

    for path in jsonl_files:
        rows = _count_jsonl_rows(path)
        signal = SIGNAL_FROM_FILE.get(path.name, path.stem)
        files[path.name] = {
            "signal": signal,
            "path": str(path),
            "rows": rows,
        }
        total_rows += rows

    return {
        "path": str(stage_dir),
        "exists": stage_dir.exists(),
        "file_count": len(jsonl_files),
        "row_count": total_rows,
        "files": files,
    }


def _summarize_signals(stage_payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    signals: dict[str, dict[str, int]] = {}
    for stage, payload in stage_payloads.items():
        files = payload.get("files", {})
        if not isinstance(files, Mapping):
            continue
        for file_payload in files.values():
            if not isinstance(file_payload, Mapping):
                continue
            signal = file_payload.get("signal")
            rows = file_payload.get("rows")
            if isinstance(signal, str) and isinstance(rows, int):
                signals.setdefault(signal, {})[f"{stage}_rows"] = rows
    return {signal: signals[signal] for signal in sorted(signals)}


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a pretrain run-level manifest from existing outputs.")
    parser.add_argument("--config", required=True, help="Path to configs/synthetic.yaml")
    parser.add_argument("--output", default=None, help="Optional manifest output path")
    parser.add_argument("--output-dir", default=None, help="Optional generated-output directory override")
    parser.add_argument("--generation-run", default=None, help="Optional generation run name override")
    args = parser.parse_args(argv)

    path = generate_run_manifest(
        config_path=args.config,
        output=args.output,
        output_dir=args.output_dir,
        generation_run=args.generation_run,
    )
    print(f"wrote pretrain run manifest to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
