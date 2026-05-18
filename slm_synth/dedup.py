from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_synth.paths import load_yaml_config, validated_dir_from_config


def canonical_record_key(obj: Any) -> str:
    """Return a stable exact-dedup key for a JSON record.

    Synthetic records often share templates and field names by design. Fuzzy
    MinHash over the whole JSON object collapses useful variation, especially
    for arithmetic and MCQ data. Exact canonical JSON matching removes only
    byte/field-order-equivalent duplicate records while preserving valid
    synthetic variation.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def minhash(text: str, num_perm: int = 128):
    # Imported lazily so exact-only dedup does not require datasketch at runtime.
    from datasketch import MinHash

    mh = MinHash(num_perm=num_perm)
    for token in text.split():
        mh.update(token.encode("utf-8"))
    return mh


def is_fuzzy_duplicate(obj: Any, seen_minhashes: list[Any], threshold: float, num_perm: int) -> bool:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    mh = minhash(text, num_perm=num_perm)
    for existing in seen_minhashes:
        if existing.jaccard(mh) >= threshold:
            return True
    seen_minhashes.append(mh)
    return False


def dedup_file(
    path: Path,
    out_path: Path,
    *,
    enable_exact: bool = True,
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 0.85,
    num_perm: int = 128,
) -> tuple[int, int, int]:
    """Deduplicate one JSONL file.

    Returns: (kept, exact_dropped, fuzzy_dropped)
    """
    seen_exact: set[str] = set()
    seen_minhashes: list[Any] = []
    kept = 0
    exact_dropped = 0
    fuzzy_dropped = 0

    with open(path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc

            if enable_exact:
                key = canonical_record_key(obj)
                if key in seen_exact:
                    exact_dropped += 1
                    continue
                seen_exact.add(key)

            if enable_fuzzy:
                if is_fuzzy_duplicate(obj, seen_minhashes, fuzzy_threshold, num_perm):
                    fuzzy_dropped += 1
                    continue

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    return kept, exact_dropped, fuzzy_dropped


def dedup_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg_dedup = cfg.get("dedup") or {}
    return {
        "enable_exact": bool(cfg_dedup.get("enable_exact", True)),
        "enable_fuzzy": bool(cfg_dedup.get("enable_fuzzy", False)),
        "fuzzy_threshold": float(cfg_dedup.get("minhash_threshold", cfg_dedup.get("fuzzy_threshold", 0.85))),
        "num_perm": int(cfg_dedup.get("num_perm", 128)),
    }


def main(validated_dir: str | Path, cfg: dict[str, Any] | None = None) -> None:
    validated_dir = Path(validated_dir)
    deduped_dir = validated_dir.parent / "deduped"

    if not validated_dir.exists():
        raise FileNotFoundError(f"validated_dir does not exist: {validated_dir}")

    settings = dedup_settings(cfg or {})

    deduped_dir.mkdir(parents=True, exist_ok=True)
    for old in deduped_dir.glob("*.jsonl"):
        old.unlink()

    files = sorted(validated_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in validated_dir: {validated_dir}")

    print(
        "[dedup] mode="
        + ("exact+fuzzy" if settings["enable_fuzzy"] else "exact")
        + f" enable_exact={settings['enable_exact']}"
        + f" enable_fuzzy={settings['enable_fuzzy']}"
    )

    total_kept = 0
    total_exact_dropped = 0
    total_fuzzy_dropped = 0

    for file in files:
        out_file = deduped_dir / file.name
        kept, exact_dropped, fuzzy_dropped = dedup_file(file, out_file, **settings)
        total_kept += kept
        total_exact_dropped += exact_dropped
        total_fuzzy_dropped += fuzzy_dropped
        print(
            f"[dedup] {file.name}: kept={kept}, "
            f"exact_dropped={exact_dropped}, fuzzy_dropped={fuzzy_dropped}"
        )

    print(
        f"[dedup] Completed kept={total_kept}, "
        f"exact_dropped={total_exact_dropped}, fuzzy_dropped={total_fuzzy_dropped}"
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate validated synthetic JSONL records.")
    parser.add_argument("validated_dir", nargs="?", help="Path to validated JSONL directory.")
    parser.add_argument("--config", default=None, help="Path to configs/synthetic.yaml.")
    parser.add_argument("--enable-fuzzy", action="store_true", help="Enable fuzzy MinHash dedup for this run.")
    parser.add_argument("--fuzzy-threshold", type=float, default=None, help="Fuzzy MinHash threshold.")
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml_config(args.config)
        validated_dir = validated_dir_from_config(args.config)
    elif args.validated_dir:
        validated_dir = Path(args.validated_dir)
    else:
        parser.error("provide either validated_dir or --config")

    cfg.setdefault("dedup", {})
    if args.enable_fuzzy:
        cfg["dedup"]["enable_fuzzy"] = True
    if args.fuzzy_threshold is not None:
        cfg["dedup"]["minhash_threshold"] = args.fuzzy_threshold

    main(validated_dir, cfg=cfg)


if __name__ == "__main__":
    cli()
