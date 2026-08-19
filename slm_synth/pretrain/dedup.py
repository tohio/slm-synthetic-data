"""Global exact and structural deduplication for pretraining records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.pretrain.record_quality import SIGNAL_FROM_FILE, iter_jsonl
from slm_synth.pretrain.report_diversity import (
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    jaccard_similarity,
    normalize_public_template_text,
    template_shingles,
)

PUBLIC_FILENAME = "pretrain.jsonl"
DEDUP_REPORT_FILENAME = "dedup_report.json"
REJECTED_DUPLICATES_FILENAME = "duplicates.jsonl"
DEFAULT_SHINGLE_SIZE = 5


def render_pretraining_text(signal: str, row: Mapping[str, Any]) -> str:
    """Render one validated structured record into its public pretraining text."""
    if signal == "arithmetic":
        steps = "\n".join(f"{index}. {step}" for index, step in enumerate(row["steps"], 1))
        return f"Question: {row['question']}\nSolution:\n{steps}\nAnswer: {row['answer']}"
    if signal == "task_code":
        plan = "\n".join(f"{index}. {step}" for index, step in enumerate(row["plan"], 1))
        return f"Task: {row['task']}\nPlan:\n{plan}\nCode:\n```python\n{row['code']}\n```"
    if signal in {"educational_qa_mcq_math", "educational_qa_mcq_general"}:
        prefix = f"Evidence: {row['evidence']}\n" if signal == "educational_qa_mcq_general" else ""
        choices = "\n".join(
            f"{chr(65 + index)}. {choice}"
            for index, choice in enumerate(row["choices"])
        )
        answer = chr(65 + int(row["correct_index"]))
        return (
            f"{prefix}Question: {row['question']}\nChoices:\n{choices}\n"
            f"Answer: {answer}\nExplanation: {row['explanation']}"
        )
    if signal == "factual_restraint":
        return f"Question: {row['question']}\nAnswer: {row['safe_answer']}"
    raise ValueError(f"unsupported pretraining signal: {signal}")


def build_public_record(signal: str, row: Mapping[str, Any]) -> dict[str, Any]:
    text = render_pretraining_text(signal, row).strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "id": f"pretrain_{digest[:24]}",
        "text": text,
        "metadata": {"signal": signal},
    }


class NearDuplicateIndex:
    """Exact Jaccard verifier with an inverted shingle candidate index."""

    def __init__(self, *, threshold: float, shingle_size: int = DEFAULT_SHINGLE_SIZE):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("near-duplicate threshold must be in (0, 1]")
        if shingle_size < 1:
            raise ValueError("shingle_size must be positive")
        self.threshold = threshold
        self.shingle_size = shingle_size
        self._shingles: list[frozenset[str]] = []
        self._records: list[dict[str, str]] = []
        self._postings: dict[str, list[int]] = defaultdict(list)

    def find(self, *, text: str, signal: str) -> tuple[dict[str, str], float] | None:
        shingles = template_shingles(
            normalize_public_template_text(text, signal),
            self.shingle_size,
        )
        candidates = {
            index
            for shingle in shingles
            for index in self._postings.get(shingle, ())
        }
        for index in sorted(candidates):
            similarity = jaccard_similarity(shingles, self._shingles[index])
            if similarity >= self.threshold:
                return self._records[index], similarity
        return None

    def add(self, *, text: str, record_id: str, signal: str) -> None:
        shingles = template_shingles(
            normalize_public_template_text(text, signal),
            self.shingle_size,
        )
        index = len(self._shingles)
        self._shingles.append(shingles)
        self._records.append({"id": record_id, "signal": signal})
        for shingle in shingles:
            self._postings[shingle].append(index)


def discover_validated_files(validated_dir: Path) -> list[tuple[str, Path]]:
    files = [
        (SIGNAL_FROM_FILE.get(path.name, path.stem), path)
        for path in sorted(validated_dir.glob("*.jsonl"))
        if path.name in SIGNAL_FROM_FILE
    ]
    if not files:
        raise FileNotFoundError(f"no validated pretraining signal files found in {validated_dir}")
    return files


def consolidate_and_deduplicate(
    *,
    validated_dir: Path,
    deduped_dir: Path,
    rejected_dir: Path,
    manifest_dir: Path,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> dict[str, Any]:
    """Create one globally deduplicated public pretraining dataset."""
    files = discover_validated_files(validated_dir)
    deduped_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for stale in deduped_dir.glob("*.jsonl"):
        stale.unlink()

    output_path = deduped_dir / PUBLIC_FILENAME
    rejected_path = rejected_dir / REJECTED_DUPLICATES_FILENAME
    index = NearDuplicateIndex(threshold=threshold, shingle_size=shingle_size)
    seen_exact: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    accepted_by_signal: Counter[str] = Counter()
    rejected_by_signal: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as output, rejected_path.open("w", encoding="utf-8") as rejected:
        for signal, path in files:
            for line_number, row, parse_issues in iter_jsonl(path):
                counts["candidate_rows"] += 1
                if parse_issues or row is None:
                    counts["invalid_rows"] += 1
                    rejected_by_signal[signal] += 1
                    _write_rejection(
                        rejected,
                        signal=signal,
                        source_path=path,
                        line_number=line_number,
                        reason="invalid_jsonl_record",
                        row=row,
                    )
                    continue

                public = build_public_record(signal, row)
                exact_key = " ".join(public["text"].casefold().split())
                if exact_key in seen_exact:
                    matched = seen_exact[exact_key]
                    counts["exact_duplicate_rows"] += 1
                    if matched["signal"] != signal:
                        counts["cross_signal_exact_duplicate_rows"] += 1
                    rejected_by_signal[signal] += 1
                    _write_rejection(
                        rejected,
                        signal=signal,
                        source_path=path,
                        line_number=line_number,
                        reason="exact_duplicate",
                        row=row,
                        matched=matched,
                        similarity=1.0,
                    )
                    continue

                match = index.find(text=public["text"], signal=signal)
                if match is not None:
                    matched, similarity = match
                    counts["near_duplicate_rows"] += 1
                    if matched["signal"] != signal:
                        counts["cross_signal_near_duplicate_rows"] += 1
                    rejected_by_signal[signal] += 1
                    _write_rejection(
                        rejected,
                        signal=signal,
                        source_path=path,
                        line_number=line_number,
                        reason="near_duplicate",
                        row=row,
                        matched=matched,
                        similarity=similarity,
                    )
                    continue

                output.write(json.dumps(public, ensure_ascii=False) + "\n")
                seen_exact[exact_key] = {"id": public["id"], "signal": signal}
                index.add(text=public["text"], record_id=public["id"], signal=signal)
                counts["accepted_rows"] += 1
                accepted_by_signal[signal] += 1

    report = {
        "schema_version": 1,
        "output_path": str(output_path),
        "rejected_path": str(rejected_path),
        "near_duplicate_threshold": threshold,
        "shingle_size": shingle_size,
        "candidate_rows": counts["candidate_rows"],
        "accepted_rows": counts["accepted_rows"],
        "invalid_rows": counts["invalid_rows"],
        "exact_duplicate_rows": counts["exact_duplicate_rows"],
        "near_duplicate_rows": counts["near_duplicate_rows"],
        "cross_signal_exact_duplicate_rows": counts[
            "cross_signal_exact_duplicate_rows"
        ],
        "cross_signal_near_duplicate_rows": counts[
            "cross_signal_near_duplicate_rows"
        ],
        "accepted_rows_by_signal": dict(sorted(accepted_by_signal.items())),
        "rejected_rows_by_signal": dict(sorted(rejected_by_signal.items())),
        "publish_ready": counts["accepted_rows"] > 0,
    }
    report_path = manifest_dir / DEDUP_REPORT_FILENAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[dedup] Completed global pretraining deduplication "
        f"candidates={counts['candidate_rows']} accepted={counts['accepted_rows']} "
        f"exact_dropped={counts['exact_duplicate_rows']} "
        f"near_dropped={counts['near_duplicate_rows']} output={output_path}"
    )
    return report


def audit_public_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> dict[str, int]:
    """Fail when a consolidated public sequence contains duplicate content."""
    index = NearDuplicateIndex(threshold=threshold, shingle_size=shingle_size)
    seen_ids: set[str] = set()
    seen_exact: dict[str, str] = {}
    count = 0
    for row in rows:
        record_id = row.get("id")
        text = row.get("text")
        metadata = row.get("metadata")
        signal = metadata.get("signal") if isinstance(metadata, Mapping) else None
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise ValueError("pretraining public data contains a missing or duplicate id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"pretraining public record {record_id} has empty text")
        if not isinstance(signal, str) or signal not in SIGNAL_FROM_FILE.values():
            raise ValueError(f"pretraining public record {record_id} has an invalid signal")
        exact_key = " ".join(text.casefold().split())
        if exact_key in seen_exact:
            raise ValueError(
                "pretraining public data contains exact duplicate records: "
                f"{seen_exact[exact_key]} and {record_id}"
            )
        match = index.find(text=text, signal=signal)
        if match is not None:
            matched, similarity = match
            raise ValueError(
                "pretraining public data contains near-duplicate records: "
                f"{matched['id']} and {record_id} similarity={similarity:.6f}"
            )
        seen_ids.add(record_id)
        seen_exact[exact_key] = record_id
        index.add(text=text, record_id=record_id, signal=signal)
        count += 1
    if count == 0:
        raise ValueError("pretraining public dataset is empty")
    return {"rows": count, "exact_duplicates": 0, "near_duplicates": 0}


def _write_rejection(
    handle: Any,
    *,
    signal: str,
    source_path: Path,
    line_number: int,
    reason: str,
    row: Mapping[str, Any] | None,
    matched: Mapping[str, str] | None = None,
    similarity: float | None = None,
) -> None:
    handle.write(
        json.dumps(
            {
                "signal": signal,
                "source_path": str(source_path),
                "line_number": line_number,
                "reason": reason,
                "matched": dict(matched or {}),
                "similarity": round(similarity, 6) if similarity is not None else None,
                "row": dict(row) if row is not None else None,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def run_from_config(config_path: str) -> dict[str, Any]:
    cfg = load_yaml_config(config_path)
    output_dir = resolve_output_dir(cfg)
    dedup_cfg = cfg.get("dedup", {}) or {}
    return consolidate_and_deduplicate(
        validated_dir=output_dir / "validated",
        deduped_dir=output_dir / "deduped",
        rejected_dir=output_dir / "rejected",
        manifest_dir=output_dir / "manifests",
        threshold=float(dedup_cfg.get("near_duplicate_threshold", DEFAULT_NEAR_DUPLICATE_THRESHOLD)),
        shingle_size=int(dedup_cfg.get("shingle_size", DEFAULT_SHINGLE_SIZE)),
    )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Globally deduplicate and consolidate validated pretraining records"
    )
    parser.add_argument("--config", required=True, help="Path to configs/synthetic.yaml")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    cli()
