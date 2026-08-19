#!/usr/bin/env python3
"""Measure structural repetition in generated pretraining records.

The report uses deterministic reservoir samples so the cost stays bounded for
production corpora while still reporting full-file row and parse-error counts.
The deduped-stage command can make the audit publication-blocking.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import keyword
import random
import re
import tokenize
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.pretrain.artifacts.educational_qa_mcq_general import (
    EducationalQAMCQGeneralArtifactFactory,
)
from slm_synth.pretrain.artifacts.factual_restraint import FactualRestraintArtifactFactory
from slm_synth.pretrain.artifacts.lexicon import (
    CITIES,
    COMPANY_NAMES,
    FIRST_NAMES,
    LAST_NAMES,
    ORGANIZATION_NAMES,
    PROJECT_NAMES,
    VENUES,
)
from slm_synth.pretrain.record_quality import SIGNAL_FROM_FILE

REPORT_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_SIZE = 10_000
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.80
DEFAULT_NUM_PERM = 64
DEFAULT_SHINGLE_SIZE = 5
DEFAULT_TOP_CLUSTERS = 20

_EXCLUDED_TEXT_FIELDS = frozenset(
    {
        "type",
        "answer",
        "correct_index",
        "verification_answer",
        "verification_expression",
    }
)
_NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,:/-]\d+)*(?![\w])")
_QUOTED_RE = re.compile(r"(?P<quote>[\"'`])[^\n\"'`]{1,120}(?P=quote)")
_CAPITALIZED_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+){1,3}\b")
_TOKEN_RE = re.compile(r"<[a-z_]+>|[a-z]+|[-+*/<>=]+|[{}()[\],.:;]")
_PYTHON_FENCE_RE = re.compile(r"```python\s*\n(?P<code>.*?)\n```", re.DOTALL | re.IGNORECASE)
_MCQ_ANSWER_RE = re.compile(r"(?im)^answer:\s*[a-d]\s*$")


def _flatten_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _flatten_values(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _flatten_values(nested)


def _controlled_slot_values() -> tuple[str, ...]:
    general = EducationalQAMCQGeneralArtifactFactory
    restraint = FactualRestraintArtifactFactory
    values: set[str] = {
        *CITIES,
        *COMPANY_NAMES,
        *ORGANIZATION_NAMES,
        *PROJECT_NAMES,
        *VENUES,
        *(f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES),
    }
    for source in (
        general.PLACES,
        general.OBJECTS,
        general.ADVERBS,
        general.VERBS,
        general.FICTIONAL_REGIONS,
        general.FICTIONAL_LABELS,
        general.DEPARTMENTS,
        general.VARIABLES,
        general.CATEGORIES,
        restraint.EVENT_TYPES,
        restraint.PRIVATE_ROLES,
        restraint.PRODUCT_TYPES,
        restraint.RUMOR_ACTIONS,
    ):
        values.update(str(item) for item in source)
    for source in (general.ADJECTIVE_CONTEXT, general.VOCABULARY_SUBJECTS):
        values.update(_flatten_values(source))
    return tuple(sorted((value for value in values if len(value) >= 3), key=len, reverse=True))


_CONTROLLED_SLOT_VALUES = _controlled_slot_values()
_CONTROLLED_SLOT_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(value) for value in _CONTROLLED_SLOT_VALUES) + r")(?!\w)",
    re.IGNORECASE,
)


def normalize_template_text(text: str) -> str:
    """Collapse known slot values while preserving the surrounding sentence frame."""
    value = unicodedata.normalize("NFKC", text)
    value = _CAPITALIZED_ENTITY_RE.sub(" <entity> ", value)
    value = _CONTROLLED_SLOT_RE.sub(" <slot> ", value)
    value = _QUOTED_RE.sub(" <quoted> ", value)
    value = _NUMBER_RE.sub(" <num> ", value)
    value = value.casefold()
    return " ".join(_TOKEN_RE.findall(value))


def normalize_code_template(code: str) -> str:
    """Normalize Python identifiers and literals while retaining program structure."""
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        normalized: list[str] = []
        for token_info in stream:
            token_type = token_info.type
            token_text = token_info.string
            if token_type == tokenize.NAME:
                normalized.append(token_text if keyword.iskeyword(token_text) else "<id>")
            elif token_type == tokenize.NUMBER:
                normalized.append("<num>")
            elif token_type == tokenize.STRING:
                normalized.append("<str>")
            elif token_type == tokenize.OP:
                normalized.append(token_text)
        return " ".join(normalized)
    except (IndentationError, tokenize.TokenError):
        return normalize_template_text(code)


def normalize_public_template_text(text: str, signal: str | None = None) -> str:
    """Normalize exported text, retaining code structure but not identifiers."""
    if signal in {"educational_qa_mcq_math", "educational_qa_mcq_general"}:
        text = _MCQ_ANSWER_RE.sub("Answer: <choice>", text)
    if signal != "task_code":
        return normalize_template_text(text)
    match = _PYTHON_FENCE_RE.search(text)
    if match is None:
        return normalize_template_text(text)
    prose = text[: match.start()] + text[match.end() :]
    return " <code> ".join(
        part
        for part in (
            normalize_template_text(prose),
            normalize_code_template(match.group("code")),
        )
        if part
    )


def record_template_text(row: Mapping[str, Any]) -> str:
    """Return a field-order-stable structural representation of one public row."""
    if isinstance(row.get("text"), str):
        metadata = row.get("metadata")
        signal = metadata.get("signal") if isinstance(metadata, Mapping) else None
        return normalize_public_template_text(
            str(row["text"]),
            signal if isinstance(signal, str) else None,
        )
    parts: list[str] = []
    for field in sorted(row):
        if field in _EXCLUDED_TEXT_FIELDS:
            continue
        value = row[field]
        if field == "code" and isinstance(value, str):
            normalized = normalize_code_template(value)
            if normalized:
                parts.append(normalized)
            continue
        for text in _flatten_values(value):
            normalized = normalize_template_text(text)
            if normalized:
                parts.append(normalized)
    return " <field> ".join(parts)


def template_fingerprint(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(record_template_text(row).encode("utf-8")).hexdigest()


def template_shingles(text: str, size: int) -> frozenset[str]:
    tokens = text.split()
    if not tokens:
        return frozenset()
    if len(tokens) < size:
        return frozenset({" ".join(tokens)})
    return frozenset(" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _minhash(shingles: Iterable[str], num_perm: int) -> MinHash:
    result = MinHash(num_perm=num_perm)
    for shingle in shingles:
        result.update(shingle.encode("utf-8"))
    return result


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _reservoir_sample(path: Path, sample_size: int, seed: str) -> tuple[int, int, list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    total = 0
    bad_json = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if not isinstance(row, dict):
                bad_json += 1
                continue
            if len(rows) < sample_size:
                rows.append(row)
                continue
            replacement = rng.randrange(total - bad_json)
            if replacement < sample_size:
                rows[replacement] = row
    return total, bad_json, rows


def _record_signal(row: Mapping[str, Any], fallback: str) -> str:
    metadata = row.get("metadata")
    signal = metadata.get("signal") if isinstance(metadata, Mapping) else None
    return signal if isinstance(signal, str) and signal else fallback


def _reservoir_samples_by_signal(
    path: Path,
    *,
    sample_size: int,
    seed_prefix: str,
    fallback_signal: str,
) -> tuple[Counter[str], int, dict[str, list[dict[str, Any]]]]:
    """Sample a consolidated file independently for every metadata signal."""
    totals: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    randomizers: dict[str, random.Random] = {}
    bad_json = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if not isinstance(row, dict):
                bad_json += 1
                continue
            signal = _record_signal(row, fallback_signal)
            totals[signal] += 1
            rows = samples[signal]
            if len(rows) < sample_size:
                rows.append(row)
                continue
            rng = randomizers.setdefault(signal, random.Random(f"{seed_prefix}:{signal}"))
            replacement = rng.randrange(totals[signal])
            if replacement < sample_size:
                rows[replacement] = row
    return totals, bad_json, dict(samples)


def _cluster_summary(
    *,
    rows: list[dict[str, Any]],
    threshold: float,
    num_perm: int,
    shingle_size: int,
    top_clusters: int,
) -> dict[str, Any]:
    templates = [record_template_text(row) for row in rows]
    template_counts = Counter(templates)
    exact_repeated_rows = sum(count for count in template_counts.values() if count > 1)
    unique_templates = sorted(template_counts)
    shingle_sets = [template_shingles(value, shingle_size) for value in unique_templates]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    disjoint = _DisjointSet(len(unique_templates))
    verified_pairs = 0
    for index, shingles in enumerate(shingle_sets):
        signature = _minhash(shingles, num_perm)
        for candidate_key in lsh.query(signature):
            candidate = int(candidate_key)
            if jaccard_similarity(shingles, shingle_sets[candidate]) >= threshold:
                disjoint.union(index, candidate)
                verified_pairs += 1
        lsh.insert(str(index), signature)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(unique_templates)):
        clusters[disjoint.find(index)].append(index)
    repeated = [
        members
        for members in clusters.values()
        if sum(template_counts[unique_templates[index]] for index in members) > 1
    ]
    repeated.sort(
        key=lambda members: (
            -sum(template_counts[unique_templates[index]] for index in members),
            members[0],
        )
    )
    near_duplicate_rows = sum(
        sum(template_counts[unique_templates[index]] for index in members)
        for members in repeated
    )
    examples = []
    for members in repeated[:top_clusters]:
        cluster_size = sum(template_counts[unique_templates[index]] for index in members)
        examples.append(
            {
                "size": cluster_size,
                "unique_template_count": len(members),
                "template_previews": [unique_templates[index][:500] for index in members[:3]],
            }
        )
    sample_count = len(rows)
    largest_cluster = (
        sum(template_counts[unique_templates[index]] for index in repeated[0])
        if repeated
        else 0
    )
    return {
        "sampled_rows": sample_count,
        "exact_template_count": len(template_counts),
        "exact_template_unique_ratio": round(len(template_counts) / sample_count, 6) if sample_count else 1.0,
        "exact_repeated_rows": exact_repeated_rows,
        "near_duplicate_threshold": threshold,
        "near_duplicate_cluster_count": len(repeated),
        "near_duplicate_rows": near_duplicate_rows,
        "near_duplicate_row_ratio": round(near_duplicate_rows / sample_count, 6) if sample_count else 0.0,
        "verified_near_duplicate_pairs": verified_pairs,
        "largest_near_duplicate_cluster": largest_cluster,
        "top_near_duplicate_clusters": examples,
    }


def _artifact_family_summary(output_dir: Path, signal: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    batch_dir = output_dir / "manifests" / "grounded" / signal / "batches"
    for path in sorted(batch_dir.glob("batch_*.json")) if batch_dir.exists() else ():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for artifact in payload.get("artifacts", []):
            if isinstance(artifact, Mapping) and isinstance(artifact.get("family"), str):
                counts[str(artifact["family"])] += 1
    total = sum(counts.values())
    return {
        "artifact_count": total,
        "family_count": len(counts),
        "family_counts": dict(sorted(counts.items())),
        "largest_family_share": round(max(counts.values(), default=0) / total, 6) if total else None,
    }


def _cross_signal_near_duplicate_summary(
    *,
    template_counts: Mapping[str, Counter[str]],
    threshold: float,
    num_perm: int,
    shingle_size: int,
    top_clusters: int,
) -> dict[str, Any]:
    """Compare unique sampled templates across signals without quadratic scans."""
    entries = [
        (signal, template, count)
        for signal in sorted(template_counts)
        for template, count in sorted(template_counts[signal].items())
    ]
    shingle_sets = [template_shingles(template, shingle_size) for _, template, _ in entries]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signal_pairs: Counter[tuple[str, str]] = Counter()
    sampled_row_pairs = 0
    examples: list[dict[str, Any]] = []
    verified_template_pairs = 0
    for index, (signal, template, count) in enumerate(entries):
        signature = _minhash(shingle_sets[index], num_perm)
        for candidate_key in lsh.query(signature):
            candidate = int(candidate_key)
            other_signal, other_template, other_count = entries[candidate]
            if other_signal == signal:
                continue
            similarity = jaccard_similarity(shingle_sets[index], shingle_sets[candidate])
            if similarity < threshold:
                continue
            pair = tuple(sorted((signal, other_signal)))
            signal_pairs[pair] += 1
            verified_template_pairs += 1
            sampled_row_pairs += count * other_count
            if len(examples) < top_clusters:
                examples.append(
                    {
                        "signals": list(pair),
                        "jaccard_similarity": round(similarity, 6),
                        "left_template_preview": other_template[:300],
                        "right_template_preview": template[:300],
                    }
                )
        lsh.insert(str(index), signature)
    return {
        "near_duplicate_threshold": threshold,
        "verified_template_pairs": verified_template_pairs,
        "sampled_row_pair_occurrences": sampled_row_pairs,
        "signal_pair_counts": {
            f"{left}::{right}": count
            for (left, right), count in sorted(signal_pairs.items())
        },
        "examples": examples,
    }


def build_diversity_report(
    *,
    output_dir: Path,
    stage: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    top_clusters: int = DEFAULT_TOP_CLUSTERS,
    publish_blocking: bool = False,
) -> dict[str, Any]:
    stage_dir = output_dir / stage
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"pretraining stage directory does not exist: {stage_dir}")

    signals: dict[str, Any] = {}
    file_count = 0
    bad_json_total = 0
    global_templates: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sampled_template_counts: dict[str, Counter[str]] = {}
    for path in sorted(stage_dir.glob("*.jsonl")):
        file_count += 1
        fallback_signal = SIGNAL_FROM_FILE.get(path.name, path.stem)
        totals, bad_json, samples = _reservoir_samples_by_signal(
            path,
            sample_size=sample_size,
            seed_prefix=f"pretrain-diversity:{stage}:{sample_size}",
            fallback_signal=fallback_signal,
        )
        bad_json_total += bad_json
        for signal, rows in sorted(samples.items()):
            template_counts = Counter(record_template_text(row) for row in rows)
            sampled_template_counts[signal] = template_counts
            for template, count in template_counts.items():
                fingerprint = hashlib.sha256(template.encode("utf-8")).hexdigest()
                global_templates[fingerprint][signal] += count
            signals[signal] = {
                "path": str(path),
                "row_count": totals[signal],
                "bad_json": bad_json,
                "sample_limit": sample_size,
                "artifact_families": _artifact_family_summary(output_dir, signal),
                **_cluster_summary(
                    rows=rows,
                    threshold=threshold,
                    num_perm=num_perm,
                    shingle_size=shingle_size,
                    top_clusters=top_clusters,
                ),
            }

    overlaps = [
        {
            "template_fingerprint": fingerprint,
            "signals": dict(sorted(counts.items())),
            "sampled_occurrences": sum(counts.values()),
        }
        for fingerprint, counts in global_templates.items()
        if len(counts) > 1
    ]
    overlaps.sort(key=lambda item: (-item["sampled_occurrences"], item["template_fingerprint"]))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stage": stage,
        "file_count": file_count,
        "bad_json": bad_json_total,
        "method": {
            "sampling": "deterministic_reservoir_per_signal",
            "sample_size_per_signal": sample_size,
            "template_normalization": "controlled_slots_numbers_entities_identifiers_and_literals",
            "near_duplicate_metric": f"jaccard_{shingle_size}_token_shingles",
            "near_duplicate_threshold": threshold,
            "minhash_permutations": num_perm,
            "publish_blocking": publish_blocking,
        },
        "signals": signals,
        "cross_signal_exact_template_overlap": {
            "overlapping_template_count": len(overlaps),
            "top_overlaps": overlaps[:top_clusters],
        },
        "cross_signal_near_duplicate_overlap": _cross_signal_near_duplicate_summary(
            template_counts=sampled_template_counts,
            threshold=threshold,
            num_perm=num_perm,
            shingle_size=shingle_size,
            top_clusters=top_clusters,
        ),
    }


def write_diversity_report(*, report: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(
    config: str,
    stage: str,
    sample_size: int,
    threshold: float,
    num_perm: int,
    shingle_size: int,
    top_clusters: int,
    require_clean: bool = False,
) -> None:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if num_perm < 16:
        raise ValueError("num_perm must be at least 16")
    if shingle_size <= 0 or top_clusters <= 0:
        raise ValueError("shingle_size and top_clusters must be positive")
    output_dir = resolve_output_dir(load_yaml_config(config))
    report = build_diversity_report(
        output_dir=output_dir,
        stage=stage,
        sample_size=sample_size,
        threshold=threshold,
        num_perm=num_perm,
        shingle_size=shingle_size,
        top_clusters=top_clusters,
        publish_blocking=require_clean,
    )
    output_path = write_diversity_report(
        report=report,
        path=output_dir / "manifests" / f"diversity_report_{stage}.json",
    )
    for signal, summary in report["signals"].items():
        print(
            f"[diversity] {signal}: rows={summary['row_count']} sampled={summary['sampled_rows']} "
            f"exact_template_unique_ratio={summary['exact_template_unique_ratio']:.4f} "
            f"near_duplicate_row_ratio={summary['near_duplicate_row_ratio']:.4f} "
            f"largest_cluster={summary['largest_near_duplicate_cluster']}"
        )
    overlap = report["cross_signal_exact_template_overlap"]
    print(f"[diversity] cross_signal_exact_template_overlap={overlap['overlapping_template_count']}")
    near_overlap = report["cross_signal_near_duplicate_overlap"]
    print(f"[diversity] cross_signal_near_duplicate_pairs={near_overlap['verified_template_pairs']}")
    print(f"[diversity] Saved report: {output_path}")
    if require_clean:
        blockers = [
            signal
            for signal, summary in report["signals"].items()
            if summary["exact_repeated_rows"] or summary["near_duplicate_rows"]
        ]
        if (
            not report["file_count"]
            or report["bad_json"]
            or not report["signals"]
            or blockers
            or overlap["overlapping_template_count"]
            or near_overlap["verified_template_pairs"]
        ):
            raise SystemExit(
                "Pretraining diversity gate failed: exact or near-duplicate records remain"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report structural diversity for a generated pretraining stage.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--stage", default="deduped", choices=["raw", "validated", "deduped"])
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--near-duplicate-threshold", type=float, default=DEFAULT_NEAR_DUPLICATE_THRESHOLD)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE)
    parser.add_argument("--top-clusters", type=int, default=DEFAULT_TOP_CLUSTERS)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    main(
        args.config,
        args.stage,
        args.sample_size,
        args.near_duplicate_threshold,
        args.num_perm,
        args.shingle_size,
        args.top_clusters,
        args.require_clean,
    )
