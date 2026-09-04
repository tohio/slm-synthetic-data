"""Generate, validate, and deduplicate pretraining data to accepted-token targets."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from slm_synth.paths import load_yaml_config, resolve_output_dir
from slm_synth.pretrain.dedup import run_from_config as deduplicate_from_config
from slm_synth.pretrain.artifacts.planning import configured_candidate_capacity
from slm_synth.pretrain.generate import (
    _grounded_token_target,
    _planned_grounded_target_rows,
    run_signal,
)
from slm_synth.pretrain.grounded import GroundedBatchStore
from slm_synth.pretrain.validate import validate_signal

DEFAULT_CHARS_PER_TOKEN = 4.0
REPORT_FILENAME = "accepted_token_report.json"


def estimate_public_text_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate tokens from public training text only, excluding ids and metadata."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return max(1, math.ceil(len(text) / chars_per_token))


def accepted_counts(path: Path, *, chars_per_token: float) -> tuple[Counter[str], Counter[str]]:
    rows: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    if not path.exists():
        return rows, tokens
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        signal = record["metadata"]["signal"]
        rows[signal] += 1
        tokens[signal] += estimate_public_text_tokens(record["text"], chars_per_token)
    return rows, tokens


def verify_completion_report(output_dir: Path, expected_signals: list[str]) -> dict[str, Any]:
    """Require a complete report with zero deficit for every configured signal."""
    path = output_dir / "manifests" / REPORT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"accepted-token completion report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    signal_reports = report.get("signals")
    if not isinstance(signal_reports, Mapping):
        raise ValueError("accepted-token completion report is missing signal results")
    missing = sorted(set(expected_signals) - set(signal_reports))
    deficits = {
        signal: int(signal_reports.get(signal, {}).get("token_deficit", -1))
        for signal in expected_signals
    }
    if report.get("status") != "complete" or report.get("publish_ready") is not True or missing or any(value != 0 for value in deficits.values()):
        raise ValueError(
            "pretraining run has not reached every accepted-token allocation: "
            f"status={report.get('status')!r} missing={missing} deficits={deficits}"
        )
    print(f"[curate] verified accepted-token completion for {len(expected_signals)} signals: {path}")
    return report


def _candidate_capacity(signal: str, mix_cfg: Mapping[str, Any]) -> int:
    return configured_candidate_capacity(signal, mix_cfg)


def _initial_candidate_plan(
    cfg: Mapping[str, Any],
    signal: str,
    *,
    capacity: int | None = None,
) -> int:
    mix_cfg = cfg["mix"][signal]
    resolved_capacity = (
        _candidate_capacity(signal, mix_cfg)
        if capacity is None
        else int(capacity)
    )
    return _planned_grounded_target_rows(
        dict(cfg),
        dict(mix_cfg),
        candidate_capacity=resolved_capacity,
    )[2]


def next_candidate_plan(
    *,
    current: int,
    accepted_tokens: int,
    target_tokens: int,
    accepted_rows: int,
    attempted_candidates: int,
    avg_tokens_per_sample: float,
    capacity: int | None,
) -> int:
    """Return a fresh-candidate attempt budget for an accepted-token deficit."""
    current = max(int(current), int(attempted_candidates))
    if accepted_tokens >= target_tokens:
        return current
    if avg_tokens_per_sample <= 0:
        raise ValueError("avg_tokens_per_sample must be positive")

    missing_tokens = target_tokens - accepted_tokens
    missing_accepted_rows = max(1, math.ceil(missing_tokens / avg_tokens_per_sample))
    observed_survival = min(
        1.0,
        max(
            0.05,
            accepted_rows / max(1, attempted_candidates),
        ),
    )
    additional_attempts = max(
        1,
        math.ceil(missing_accepted_rows / observed_survival),
    )
    requested = current + additional_attempts
    if capacity is not None:
        requested = min(requested, capacity)
    return max(current, requested)


def _total_cost(output_dir: Path, signals: list[str]) -> float:
    return sum(
        float(GroundedBatchStore(output_dir, signal).telemetry_summary().get("cost", 0.0) or 0.0)
        for signal in signals
    )


def _write_report(
    *,
    output_dir: Path,
    targets: Mapping[str, int],
    accepted_rows: Mapping[str, int],
    accepted_tokens: Mapping[str, int],
    plans: Mapping[str, int],
    capacities: Mapping[str, int | None],
    chars_per_token: float,
    cost: float,
    status: str,
    stop_reason: str,
) -> dict[str, Any]:
    signals: dict[str, Any] = {}
    for signal, target in targets.items():
        accepted = int(accepted_tokens.get(signal, 0))
        capacity = capacities[signal]
        signals[signal] = {
            "target_accepted_tokens": target,
            "accepted_tokens": accepted,
            "token_deficit": max(0, target - accepted),
            "accepted_rows": int(accepted_rows.get(signal, 0)),
            "attempted_candidates": GroundedBatchStore(output_dir, signal).next_candidate_index(),
            "planned_candidate_attempts": plans[signal],
            "candidate_capacity": capacity,
            "candidate_inventory_exhausted": capacity is not None and plans[signal] >= capacity,
        }
    report = {
        "schema_version": 1,
        "status": status,
        "publish_ready": status == "complete",
        "stop_reason": stop_reason,
        "token_estimator": {"source": "public_text", "chars_per_token": chars_per_token},
        "target_accepted_tokens": sum(targets.values()),
        "accepted_tokens": sum(int(accepted_tokens.get(signal, 0)) for signal in targets),
        "token_deficit": sum(max(0, target - int(accepted_tokens.get(signal, 0))) for signal, target in targets.items()),
        "cost_usd": round(cost, 8),
        "signals": signals,
    }
    path = output_dir / "manifests" / REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[curate] "
        f"status={status} accepted_tokens={report['accepted_tokens']}/{report['target_accepted_tokens']} "
        f"deficit={report['token_deficit']} stop_reason={stop_reason} report={path}"
    )
    return report


def curate_to_accepted_token_target(
    config_path: str,
    *,
    signal_override: str | None = None,
    allow_shortfall: bool = False,
) -> dict[str, Any]:
    """Run replacement rounds until accepted public text reaches every signal target."""
    cfg = load_yaml_config(config_path)
    output_dir = resolve_output_dir(cfg)
    signals = [signal_override] if signal_override else list(cfg.get("mix", {}))
    if not signals or any(signal not in cfg.get("mix", {}) for signal in signals):
        raise ValueError(f"Unknown or empty pretraining signal selection: {signal_override!r}")

    generation_cfg = cfg.get("generation", {}) or {}
    chars_per_token = float(generation_cfg.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN))
    max_cost = generation_cfg.get("max_cost_usd")
    max_cost = float(max_cost) if max_cost is not None else None
    targets = {signal: _grounded_token_target(cfg, cfg["mix"][signal]) for signal in signals}
    capacities = {
        signal: _candidate_capacity(signal, cfg["mix"][signal])
        for signal in signals
    }
    existing_rows, existing_tokens = accepted_counts(
        output_dir / "deduped" / "pretrain.jsonl",
        chars_per_token=chars_per_token,
    )
    plans: dict[str, int] = {}
    for signal in signals:
        store = GroundedBatchStore(output_dir, signal)
        attempted_candidates = store.next_candidate_index()
        initial_plan = max(
            _initial_candidate_plan(cfg, signal, capacity=capacities[signal]),
            attempted_candidates,
        )
        mix_cfg = cfg["mix"][signal]
        if attempted_candidates == 0:
            plans[signal] = initial_plan
        else:
            plans[signal] = next_candidate_plan(
                current=initial_plan,
                accepted_tokens=int(existing_tokens.get(signal, 0)),
                target_tokens=targets[signal],
                accepted_rows=int(existing_rows.get(signal, 0)),
                attempted_candidates=attempted_candidates,
                avg_tokens_per_sample=float(
                    mix_cfg.get(
                        "avg_tokens_per_sample",
                        generation_cfg.get("avg_tokens_per_sample", 100),
                    )
                ),
                capacity=capacities[signal],
            )

    while True:
        round_cfg = copy.deepcopy(cfg)
        for signal in signals:
            round_cfg["mix"][signal]["samples"] = plans[signal]
            run_signal(signal, round_cfg, output_dir)

        raw_dir = output_dir / "raw"
        validated_dir = output_dir / "validated"
        rejected_dir = output_dir / "rejected"
        for signal in signals:
            validate_signal(raw_dir, validated_dir, rejected_dir, signal)
        deduplicate_from_config(config_path)

        rows, tokens = accepted_counts(
            output_dir / "deduped" / "pretrain.jsonl",
            chars_per_token=chars_per_token,
        )
        cost = _total_cost(output_dir, signals)
        if all(tokens.get(signal, 0) >= target for signal, target in targets.items()):
            return _write_report(
                output_dir=output_dir, targets=targets, accepted_rows=rows,
                accepted_tokens=tokens, plans=plans, capacities=capacities,
                chars_per_token=chars_per_token, cost=cost,
                status="complete", stop_reason="accepted_token_target_reached",
            )

        if max_cost is not None and cost >= max_cost:
            reason = "cost_limit_reached"
        else:
            next_plans: dict[str, int] = {}
            for signal in signals:
                mix_cfg = cfg["mix"][signal]
                store = GroundedBatchStore(output_dir, signal)
                next_plans[signal] = next_candidate_plan(
                    current=plans[signal],
                    accepted_tokens=int(tokens.get(signal, 0)),
                    target_tokens=targets[signal],
                    accepted_rows=int(rows.get(signal, 0)),
                    attempted_candidates=store.next_candidate_index(),
                    avg_tokens_per_sample=float(
                        mix_cfg.get(
                            "avg_tokens_per_sample",
                            generation_cfg.get("avg_tokens_per_sample", 100),
                        )
                    ),
                    capacity=capacities[signal],
                )
            if next_plans == plans:
                reason = "unique_candidate_inventory_exhausted"
            else:
                plans = next_plans
                continue

        report = _write_report(
            output_dir=output_dir, targets=targets, accepted_rows=rows,
            accepted_tokens=tokens, plans=plans, capacities=capacities,
            chars_per_token=chars_per_token, cost=cost,
            status="shortfall", stop_reason=reason,
        )
        if not allow_shortfall:
            raise SystemExit(
                "Pretraining accepted-token target was not reached. "
                f"deficit={report['token_deficit']} reason={reason}; see the accepted-token report."
            )
        return report


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate replacement candidates until accepted public pretraining tokens reach target"
    )
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--signal", default=None)
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        cfg = load_yaml_config(args.config)
        signals = [args.signal] if args.signal else list(cfg.get("mix", {}))
        verify_completion_report(resolve_output_dir(cfg), signals)
        return
    curate_to_accepted_token_target(
        args.config,
        signal_override=args.signal,
        allow_shortfall=args.allow_shortfall,
    )


if __name__ == "__main__":
    cli()
