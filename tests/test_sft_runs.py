import json

from slm_synth.sft.runs import (
    default_dataset_path,
    default_manifest_path,
    materialize_seed_dataset,
)


def test_default_paths():
    assert str(default_dataset_path(output_dir="data/sft", family="answer_only_arithmetic")) == (
        "data/sft/answer_only_arithmetic.jsonl"
    )
    assert str(
        default_manifest_path(
            manifest_dir="data/sft/manifests",
            family="answer_only_arithmetic",
            generation_run="sft-smoke-001",
        )
    ) == "data/sft/manifests/answer_only_arithmetic.sft-smoke-001.manifest.json"


def test_materialize_seed_dataset_writes_sft_jsonl_and_manifest(tmp_path):
    result = materialize_seed_dataset(
        family="answer_only_arithmetic",
        count=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        generation_run="sft-smoke-001",
        metadata={"source": "unit-test"},
    )

    assert result.family == "answer_only_arithmetic"
    assert result.generation_run == "sft-smoke-001"
    assert result.row_count == 2
    assert result.dataset_path == tmp_path / "datasets" / "answer_only_arithmetic.jsonl"
    assert result.manifest_path == tmp_path / "manifests" / "answer_only_arithmetic.sft-smoke-001.manifest.json"

    rows = [json.loads(line) for line in result.dataset_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [
        "sft_answer_only_arithmetic_000001",
        "sft_answer_only_arithmetic_000002",
    ]
    assert rows[0]["messages"][1]["content"] == "7"
    assert rows[0]["metadata"]["category"] == "answer_only_compliance"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_type"] == "sft"
    assert manifest["row_count"] == 2
    assert manifest["categories"] == {"answer_only_compliance": 2}
    assert manifest["eval_families"] == {"basic_arithmetic_qa": 2}
    assert manifest["metadata"] == {
        "family": "answer_only_arithmetic",
        "start_index": 1,
        "source": "unit-test",
    }


def test_materialize_seed_dataset_supports_custom_filenames(tmp_path):
    result = materialize_seed_dataset(
        family="answer_only_arithmetic",
        count=1,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        generation_run="sft-smoke-001",
        dataset_filename="sft.jsonl",
        manifest_filename="sft.manifest.json",
        start_index=5,
    )

    assert result.dataset_path == tmp_path / "datasets" / "sft.jsonl"
    assert result.manifest_path == tmp_path / "manifests" / "sft.manifest.json"

    row = json.loads(result.dataset_path.read_text(encoding="utf-8"))
    assert row["id"] == "sft_answer_only_arithmetic_000005"
