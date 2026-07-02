import json

from slm_synth.dpo.runs import (
    default_dataset_path,
    default_manifest_path,
    materialize_seed_dataset,
    materialize_seed_run,
    resolve_seed_families,
)
from slm_synth.dpo.seeds import DPO_SEED_FAMILIES


def test_default_paths():
    assert str(default_dataset_path(output_dir="data/dpo", family="answer_only_arithmetic")) == (
        "data/dpo/answer_only_arithmetic.jsonl"
    )
    assert str(
        default_manifest_path(
            manifest_dir="data/dpo/manifests",
            family="answer_only_arithmetic",
            generation_run="dpo-smoke-001",
        )
    ) == "data/dpo/manifests/answer_only_arithmetic.dpo-smoke-001.manifest.json"


def test_materialize_seed_dataset_writes_dpo_jsonl_and_manifest(tmp_path):
    result = materialize_seed_dataset(
        family="answer_only_arithmetic",
        count=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        generation_run="dpo-smoke-001",
        metadata={"source": "unit-test"},
    )

    assert result.family == "answer_only_arithmetic"
    assert result.generation_run == "dpo-smoke-001"
    assert result.row_count == 2
    assert result.dataset_path == tmp_path / "datasets" / "answer_only_arithmetic.jsonl"
    assert result.manifest_path == tmp_path / "manifests" / "answer_only_arithmetic.dpo-smoke-001.manifest.json"

    rows = [json.loads(line) for line in result.dataset_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [
        "dpo_answer_only_arithmetic_000001",
        "dpo_answer_only_arithmetic_000002",
    ]
    assert rows[0]["chosen"][0]["content"] == "7"
    assert rows[0]["rejected"][0]["content"] == "The answer is 7 because 3 plus 4 equals 7."
    assert rows[0]["metadata"]["failure_mode"] == "extra_explanation"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_type"] == "dpo"
    assert manifest["row_count"] == 2
    assert manifest["categories"] == {"answer_only_compliance": 2}
    assert manifest["eval_families"] == {"basic_arithmetic_qa": 2}
    assert manifest["failure_modes"] == {"extra_explanation": 2}
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
        generation_run="dpo-smoke-001",
        dataset_filename="dpo.jsonl",
        manifest_filename="dpo.manifest.json",
        start_index=5,
    )

    assert result.dataset_path == tmp_path / "datasets" / "dpo.jsonl"
    assert result.manifest_path == tmp_path / "manifests" / "dpo.manifest.json"

    row = json.loads(result.dataset_path.read_text(encoding="utf-8"))
    assert row["id"] == "dpo_answer_only_arithmetic_000005"


def test_resolve_seed_families_defaults_to_all_dpo_families():
    assert resolve_seed_families(None) == tuple(sorted(DPO_SEED_FAMILIES))
    assert resolve_seed_families(["all"]) == tuple(sorted(DPO_SEED_FAMILIES))


def test_resolve_seed_families_rejects_all_with_explicit_dpo_families():
    try:
        resolve_seed_families(["all", "answer_only_arithmetic"])
    except ValueError as exc:
        assert "'all' cannot be combined" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_seed_families_rejects_duplicate_dpo_families():
    try:
        resolve_seed_families(["answer_only_arithmetic", "answer_only_arithmetic"])
    except ValueError as exc:
        assert "Duplicate DPO seed family" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_materialize_seed_run_writes_one_dpo_dataset_per_family(tmp_path):
    result = materialize_seed_run(
        families=["answer_only_arithmetic", "repeat_exact_n_times"],
        count_per_family=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        generation_run="dpo-smoke-001",
        start_index=3,
    )

    assert result.families == ("answer_only_arithmetic", "repeat_exact_n_times")
    assert result.generation_run == "dpo-smoke-001"
    assert result.row_count == 4
    assert result.manifest_path == tmp_path / "manifests" / "dpo-smoke-001.manifest.json"
    assert [item.family for item in result.results] == ["answer_only_arithmetic", "repeat_exact_n_times"]
    assert (tmp_path / "datasets" / "answer_only_arithmetic.jsonl").exists()
    assert (tmp_path / "datasets" / "repeat_exact_n_times.jsonl").exists()
    assert (tmp_path / "manifests" / "answer_only_arithmetic.dpo-smoke-001.manifest.json").exists()
    assert (tmp_path / "manifests" / "repeat_exact_n_times.dpo-smoke-001.manifest.json").exists()

    row = json.loads((tmp_path / "datasets" / "answer_only_arithmetic.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["id"] == "dpo_answer_only_arithmetic_000003"

    run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["dataset_type"] == "dpo"
    assert run_manifest["families"] == ["answer_only_arithmetic", "repeat_exact_n_times"]
    assert run_manifest["total_rows"] == 4
    assert run_manifest["metadata"] == {
        "family_count": 2,
        "count_per_family": 2,
        "start_index": 3,
    }


def test_materialize_seed_run_supports_custom_dpo_run_manifest_filename(tmp_path):
    result = materialize_seed_run(
        families=["answer_only_arithmetic"],
        count_per_family=1,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        generation_run="dpo-smoke-001",
        run_manifest_filename="custom.dpo.run.json",
    )

    assert result.manifest_path == tmp_path / "manifests" / "custom.dpo.run.json"
    assert result.manifest_path.exists()
