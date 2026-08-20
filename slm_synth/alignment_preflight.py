"""Offline quality gate for the complete generic SFT and DPO inventories."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_synth.dpo.source_catalog import DPO_SOURCE_CATALOG
from slm_synth.dpo.spec_builders import build_complete_inventory as build_complete_dpo_inventory
from slm_synth.dpo.specs import dpo_source_fingerprint, require_unique_dpo_sources
from slm_synth.sft.source_catalog import SFT_SOURCE_CATALOG
from slm_synth.sft.spec_builders import build_complete_inventory as build_complete_sft_inventory
from slm_synth.sft.specs import require_unique_sft_sources, sft_source_fingerprint
from slm_synth.taxonomy import CONTEXT_MODES, OUTPUT_MODES, PREFERENCE_DIMENSIONS, TASK_FAMILIES

MIN_SOURCES_PER_GROUP = 5
NEAR_DUPLICATE_THRESHOLD = 0.88
MAX_TEMPLATE_SHARE = 0.40
_SUPPORTED_SFT_OUTPUT_MODES = frozenset(OUTPUT_MODES - {"tool_call"})
_NUMBER_RE = re.compile(r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def preflight_sft_inventory() -> dict[str, Any]:
    """Validate every SFT source, regardless of the requested generation slice."""
    _require_catalog_keys(SFT_SOURCE_CATALOG, TASK_FAMILIES, "SFT task family")
    _validate_declared_sources(SFT_SOURCE_CATALOG, kind="SFT")
    specs = build_complete_sft_inventory()
    require_unique_sft_sources(specs)
    _reject_near_duplicate_specs(specs, sft_source_fingerprint, kind="SFT")
    _require_sft_axis_coverage(specs)
    return _report("sft", SFT_SOURCE_CATALOG, specs)


def preflight_dpo_inventory() -> dict[str, Any]:
    """Validate every independent DPO source and its preference coverage."""
    _require_catalog_keys(DPO_SOURCE_CATALOG, PREFERENCE_DIMENSIONS, "DPO preference dimension")
    _validate_declared_sources(DPO_SOURCE_CATALOG, kind="DPO")
    specs = build_complete_dpo_inventory()
    require_unique_dpo_sources(specs)
    _reject_near_duplicate_specs(specs, dpo_source_fingerprint, kind="DPO")
    dimensions = Counter(spec["metadata"]["preference_dimension"] for spec in specs)
    if set(dimensions) != set(PREFERENCE_DIMENSIONS):
        raise ValueError("DPO inventory does not cover every preference dimension")
    _require_dpo_is_independent(specs)
    return _report("dpo", DPO_SOURCE_CATALOG, specs)


def preflight_all_inventories() -> dict[str, Any]:
    """Validate SFT, DPO, and their source separation."""
    sft = preflight_sft_inventory()
    dpo = preflight_dpo_inventory()
    return {"status": "clean", "sft": sft, "dpo": dpo}


def _require_catalog_keys(catalog: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    actual = set(catalog)
    expected_set = set(expected)
    if actual != expected_set:
        raise ValueError(f"{label} catalog mismatch: missing={sorted(expected_set - actual)}, extra={sorted(actual - expected_set)}")


def _validate_declared_sources(catalog: Mapping[str, tuple[dict[str, Any], ...]], *, kind: str) -> None:
    seen_keys: dict[str, str] = {}
    seen_surfaces: dict[str, str] = {}
    for group, sources in catalog.items():
        if len(sources) < MIN_SOURCES_PER_GROUP:
            raise ValueError(f"{kind} {group!r} has {len(sources)} sources; minimum is {MIN_SOURCES_PER_GROUP}")
        template_counts = Counter(str(source.get("metadata", {}).get("template_family", "")) for source in sources)
        template, count = template_counts.most_common(1)[0]
        if count / len(sources) > MAX_TEMPLATE_SHARE:
            raise ValueError(f"{kind} {group!r} is concentrated in template {template!r}: {count}/{len(sources)}")
        for source in sources:
            key = source.get("source_key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{kind} {group!r} contains a source without source_key")
            if re.search(r"(?:_|-)(?:variant|version|style|tone)?\d+$", key):
                raise ValueError(f"{kind} source_key {key!r} looks like a superficial numbered variant")
            prior = seen_keys.get(key)
            if prior is not None:
                raise ValueError(f"{kind} source_key {key!r} repeats across {prior!r} and {group!r}")
            seen_keys[key] = group
            surface = _superficial_fingerprint(source)
            prior_surface = seen_surfaces.get(surface)
            if prior_surface is not None:
                raise ValueError(
                    f"{kind} sources {prior_surface!r} and {key!r} differ only by numbers, formatting, or metadata"
                )
            seen_surfaces[surface] = key


def _superficial_fingerprint(source: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in source.items() if key not in {"source_key", "metadata", "holdout_key"}}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False).lower()
    text = _NUMBER_RE.sub("<number>", text)
    return " ".join(_TOKEN_RE.findall(text))


def _reject_near_duplicate_specs(specs: list[dict[str, Any]], fingerprint, *, kind: str) -> None:
    normalized = [(spec["id"], fingerprint(spec)) for spec in specs]
    for index, (left_id, left) in enumerate(normalized):
        for right_id, right in normalized[index + 1 :]:
            ratio = _token_jaccard(left, right)
            if ratio >= NEAR_DUPLICATE_THRESHOLD:
                raise ValueError(f"{kind} near-duplicate sources: {left_id}/{right_id} similarity={ratio:.3f}")


def _require_sft_axis_coverage(specs: list[dict[str, Any]]) -> None:
    interactions = {mode for spec in specs for mode in spec["metadata"]["interaction_modes"]}
    required_interactions = {"single_turn", "multi_turn", "system_conditioned"}
    if not required_interactions <= interactions:
        raise ValueError(f"SFT inventory missing interaction coverage: {sorted(required_interactions - interactions)}")
    contexts = {spec["metadata"]["context_mode"] for spec in specs}
    if contexts != set(CONTEXT_MODES):
        raise ValueError(f"SFT inventory context coverage mismatch: missing={sorted(set(CONTEXT_MODES) - contexts)}")
    outputs = {spec["metadata"]["output_mode"] for spec in specs}
    if not _SUPPORTED_SFT_OUTPUT_MODES <= outputs:
        raise ValueError(f"SFT inventory missing supported output coverage: {sorted(_SUPPORTED_SFT_OUTPUT_MODES - outputs)}")


def _require_dpo_is_independent(dpo_specs: list[dict[str, Any]]) -> None:
    sft_specs = build_complete_sft_inventory()
    sft_text = {_task_text(spec) for spec in sft_specs}
    for spec in dpo_specs:
        text = _task_text(spec)
        if text in sft_text:
            raise ValueError(f"DPO source {spec['id']} copies an SFT prompt")
        for sft_spec in sft_specs:
            ratio = _token_jaccard(text, _task_text(sft_spec))
            if ratio >= NEAR_DUPLICATE_THRESHOLD:
                raise ValueError(f"DPO source {spec['id']} is a near-copy of SFT source {sft_spec['id']}")


def _task_text(spec: Mapping[str, Any]) -> str:
    payload = {"instruction": spec["instruction"], "variables": spec.get("variables", {})}
    return " ".join(_TOKEN_RE.findall(json.dumps(payload, sort_keys=True, ensure_ascii=False).lower()))


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _report(kind: str, catalog: Mapping[str, tuple[dict[str, Any], ...]], specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": kind,
        "status": "clean",
        "total_capacity": len(specs),
        "capacity_by_group": {group: len(sources) for group, sources in sorted(catalog.items())},
        "interaction_modes": dict(sorted(Counter(mode for spec in specs for mode in spec["metadata"]["interaction_modes"]).items())),
        "output_modes": dict(sorted(Counter(spec["metadata"]["output_mode"] for spec in specs).items())),
        "context_modes": dict(sorted(Counter(spec["metadata"]["context_mode"] for spec in specs).items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight complete generic alignment source inventories.")
    parser.add_argument("--kind", choices=["all", "sft", "dpo"], default="all")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = (
        preflight_all_inventories()
        if args.kind == "all"
        else preflight_sft_inventory()
        if args.kind == "sft"
        else preflight_dpo_inventory()
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        print(f"[alignment-preflight] wrote {path}")
    print(f"[alignment-preflight] {args.kind}: clean")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
