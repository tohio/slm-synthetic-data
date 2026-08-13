import json
import re

import pytest

from slm_synth.distillation_sft.adjudication_backfill import backfill_adjudicated_run
from slm_synth.distillation_sft.orchestration import generate_prompt_spec_multi_signal_run


class ArithmeticBackend:
    def __init__(self):
        self.requested_ids = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        items = json.loads(prompt.split("Input items:\n", 1)[1])["items"]
        self.requested_ids.extend(item["id"] for item in items)
        responses = []
        for item in items:
            match = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", item["prompt"])
            assert match is not None
            left = int(match.group(1))
            right = int(match.group(3))
            operation = match.group(2)
            answer = {
                "+": left + right,
                "-": left - right,
                "*": left * right,
                "/": left // right,
            }[operation]
            responses.append(
                {"id": item["id"], "reasoning": None, "response": str(answer)}
            )
        return {
            "data": {"items": responses},
            "telemetry": {
                "usage": {
                    "prompt_tokens": len(items),
                    "completion_tokens": len(items),
                    "total_tokens": len(items) * 2,
                    "cost": 0.01,
                }
            },
        }


def _build_run(tmp_path):
    run_dir = tmp_path / "distillation-sft-target-001"
    backend = ArithmeticBackend()
    generate_prompt_spec_multi_signal_run(
        signals=["arithmetic"],
        count_per_signal=3,
        output_dir=run_dir / "datasets",
        manifest_dir=run_dir / "manifests",
        teacher_model="test/teacher",
        generation_run=run_dir.name,
        max_tokens=256,
        batch_size=3,
        backend_factory=lambda signal: backend,
    )
    return run_dir


def test_backfill_replaces_only_adjudication_deficit_and_updates_manifests(tmp_path):
    run_dir = _build_run(tmp_path)
    dataset = run_dir / "datasets" / "arithmetic.jsonl"
    original_rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    rejected_row = original_rows[1]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in (original_rows[0], original_rows[2])),
        encoding="utf-8",
    )
    rejected_dir = run_dir / "rejected"
    rejected_dir.mkdir()
    (rejected_dir / "repeated_response_adjudications.jsonl").write_text(
        json.dumps({"signal": "arithmetic", "row": rejected_row}) + "\n",
        encoding="utf-8",
    )
    backend = ArithmeticBackend()

    result = backfill_adjudicated_run(
        run_dir=run_dir,
        max_tokens=256,
        batch_size=1,
        concurrency=1,
        backend_factory=lambda signal: backend,
    )

    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    assert result["added_rows"] == 1
    assert backend.requested_ids == ["arithmetic-000004"]
    assert [row["id"] for row in rows] == [
        "arithmetic-000001",
        "arithmetic-000003",
        "arithmetic-000004",
    ]
    run_manifest = json.loads(
        (run_dir / "manifests" / f"{run_dir.name}.manifest.json").read_text(encoding="utf-8")
    )
    signal_manifest = json.loads(
        (run_dir / "manifests" / f"arithmetic.{run_dir.name}.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_manifest["total_rows"] == 3
    assert run_manifest["datasets"][0]["row_count"] == 3
    assert run_manifest["metadata"]["publish_ready"] is True
    assert run_manifest["metadata"]["planned_prompt_rows"] == 4
    assert run_manifest["metadata"]["rejected_rows"] == 1
    assert run_manifest["metadata"]["rejection_reasons"] == {
        "repeated_response_cluster_adjudication": 1
    }
    assert run_manifest["metadata"]["llm_telemetry"]["usage"]["cost"] == 0.02
    assert signal_manifest["row_count"] == 3
    assert signal_manifest["metadata"]["accepted_target"]["remaining"] == 0
    assert signal_manifest["metadata"]["adjudication_backfills"][0]["start_index"] == 4


def test_backfill_complete_run_does_not_construct_backend(tmp_path):
    run_dir = _build_run(tmp_path)

    def fail_backend(signal):
        raise AssertionError("complete run must not call the provider backend")

    result = backfill_adjudicated_run(
        run_dir=run_dir,
        max_tokens=256,
        batch_size=1,
        concurrency=1,
        backend_factory=fail_backend,
    )

    assert result["added_rows"] == 0
    assert result["rows"] == 3


def test_backfill_rejects_unexplained_deficit_before_constructing_backend(tmp_path):
    run_dir = _build_run(tmp_path)
    dataset = run_dir / "datasets" / "arithmetic.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:-1]),
        encoding="utf-8",
    )

    def fail_backend(signal):
        raise AssertionError("unexplained deficit must fail before provider setup")

    with pytest.raises(ValueError, match="cannot backfill unexplained arithmetic deficit"):
        backfill_adjudicated_run(
            run_dir=run_dir,
            max_tokens=256,
            batch_size=1,
            concurrency=1,
            backend_factory=fail_backend,
        )
