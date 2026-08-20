import json

import pytest

from slm_synth.pretrain import curate


def _public_row(signal: str, text: str, index: int) -> dict:
    return {"id": f"row-{index}", "text": text, "metadata": {"signal": signal}}


def test_accepted_counts_use_public_text_only(tmp_path):
    path = tmp_path / "pretrain.jsonl"
    rows = [
        _public_row("arithmetic", "a" * 8, 1),
        _public_row("arithmetic", "b" * 9, 2),
        _public_row("task_code", "c" * 4, 3),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    counts, tokens = curate.accepted_counts(path, chars_per_token=4.0)

    assert counts == {"arithmetic": 2, "task_code": 1}
    assert tokens == {"arithmetic": 5, "task_code": 1}


def test_next_candidate_plan_adds_replacements_and_respects_capacity():
    assert curate.next_candidate_plan(
        current=32,
        accepted_tokens=700,
        target_tokens=1000,
        avg_tokens_per_sample=100,
        capacity=96,
    ) == 35
    assert curate.next_candidate_plan(
        current=64,
        accepted_tokens=700,
        target_tokens=1000,
        avg_tokens_per_sample=100,
        capacity=70,
    ) == 67


def test_curator_generates_replacements_until_accepted_target(monkeypatch, tmp_path):
    output_dir = tmp_path / "run"
    config = tmp_path / "synthetic.yaml"
    config.write_text(
        f'''output_dir: "{output_dir}"
target_total_tokens: 10
backend: {{model: test}}
generation:
  batch_size: 2
  chars_per_token: 1
mix:
  arithmetic:
    architecture: grounded
    share: 1.0
    avg_tokens_per_sample: 5
    batch_size: 2
    max_unique_candidates: 6
''',
        encoding="utf-8",
    )
    requested = []

    def fake_run_signal(signal, cfg, run_dir):
        requested.append(cfg["mix"][signal]["samples"])
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        (run_dir / "raw" / "arithmetic.jsonl").write_text("{}\n", encoding="utf-8")

    accepted_rounds = [
        [_public_row("arithmetic", "123456", 1)],
        [_public_row("arithmetic", "123456", 1), _public_row("arithmetic", "abcdef", 2)],
    ]
    calls = {"round": 0}

    monkeypatch.setattr(curate, "run_signal", fake_run_signal)
    monkeypatch.setattr(curate, "validate_signal", lambda *args, **kwargs: (1, 0))

    def fake_dedup(_config):
        rows = accepted_rounds[calls["round"]]
        calls["round"] += 1
        path = output_dir / "deduped" / "pretrain.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    monkeypatch.setattr(curate, "deduplicate_from_config", fake_dedup)
    monkeypatch.setattr(curate, "_total_cost", lambda *args, **kwargs: 0.0)

    report = curate.curate_to_accepted_token_target(str(config))

    assert requested == [2, 3]
    assert report["status"] == "complete"
    assert report["accepted_tokens"] == 12


def test_curator_reports_inventory_shortfall_instead_of_silent_completion(monkeypatch, tmp_path):
    output_dir = tmp_path / "run"
    config = tmp_path / "synthetic.yaml"
    config.write_text(
        f'''output_dir: "{output_dir}"
target_total_tokens: 20
backend: {{model: test}}
generation:
  batch_size: 2
  chars_per_token: 1
mix:
  arithmetic:
    architecture: grounded
    share: 1.0
    avg_tokens_per_sample: 5
    batch_size: 2
    max_unique_candidates: 2
''',
        encoding="utf-8",
    )

    def fake_run_signal(signal, cfg, run_dir):
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        (run_dir / "raw" / "arithmetic.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(curate, "run_signal", fake_run_signal)
    monkeypatch.setattr(curate, "validate_signal", lambda *args, **kwargs: (1, 0))

    def fake_dedup(_config):
        path = output_dir / "deduped" / "pretrain.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_public_row("arithmetic", "12345", 1)) + "\n", encoding="utf-8")

    monkeypatch.setattr(curate, "deduplicate_from_config", fake_dedup)
    monkeypatch.setattr(curate, "_total_cost", lambda *args, **kwargs: 0.0)

    with pytest.raises(SystemExit, match="accepted-token target was not reached"):
        curate.curate_to_accepted_token_target(str(config))

    report = json.loads((output_dir / "manifests" / curate.REPORT_FILENAME).read_text())
    assert report["status"] == "shortfall"
    assert report["stop_reason"] == "unique_candidate_inventory_exhausted"
    assert report["accepted_tokens"] == 5
    assert report["token_deficit"] == 15


def test_verify_completion_requires_zero_deficit_for_every_signal(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    path = manifest_dir / curate.REPORT_FILENAME
    path.write_text(json.dumps({
        "status": "complete",
        "publish_ready": True,
        "signals": {
            "arithmetic": {"token_deficit": 0},
            "task_code": {"token_deficit": 4},
        },
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="every accepted-token allocation"):
        curate.verify_completion_report(tmp_path, ["arithmetic", "task_code"])

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["signals"]["task_code"]["token_deficit"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert curate.verify_completion_report(tmp_path, ["arithmetic", "task_code"])["status"] == "complete"
