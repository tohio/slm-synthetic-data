import json

import pytest

from slm_synth.accepted_target import UnderfilledRunError
from slm_synth.distillation_dpo.acceptance import (
    build_dataset_acceptance_report,
    require_dataset_acceptance,
)
from slm_synth.distillation_dpo.pair_quality import filter_pairs_by_quality
from slm_synth.distillation_dpo.runs import generate_llm_run
from slm_synth.distillation_dpo.spec_builders import build_production_rows


FAMILY = "teacher_response_preference"


def _with_id(row, row_id):
    value = dict(row)
    value["id"] = row_id
    return value


def test_duplicate_prompts_and_triples_do_not_count_as_accepted_pairs():
    original = build_production_rows(family=FAMILY, count=1)[0]
    same_prompt = _with_id(original, "same-prompt")
    same_prompt["chosen"] = [{"role": "assistant", "content": "A different chosen response."}]
    same_prompt["rejected"] = [{"role": "assistant", "content": "A different rejected response."}]
    same_triple = _with_id(original, "same-triple")

    accepted, summary = filter_pairs_by_quality(
        family=FAMILY,
        rows=[original, same_prompt, same_triple],
    )

    assert [row["id"] for row in accepted] == [original["id"]]
    assert summary.checked_pairs == 3
    assert summary.accepted_pairs == 1
    assert summary.rejected_pairs == 2
    assert summary.duplicate_prompt_pairs == 2
    assert summary.duplicate_triple_pairs == 1
    assert summary.rejection_reasons == {
        "duplicate_preference_triple": 1,
        "duplicate_prompt": 2,
    }


def test_acceptance_report_blocks_duplicate_public_rows():
    original = build_production_rows(family=FAMILY, count=1)[0]
    duplicate = _with_id(original, "duplicate-triple")

    report = build_dataset_acceptance_report([original, duplicate])

    assert report["accepted_pairs"] == 1
    assert report["unique_prompt_count"] == 1
    assert report["unique_triple_count"] == 1
    assert report["uniqueness_satisfied"] is False
    assert report["publish_ready"] is False
    with pytest.raises(ValueError, match="dataset acceptance contract"):
        require_dataset_acceptance(report, artifact_name="test artifact")


class _RepeatingBackend:
    def __init__(self):
        self.first = None

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        request = json.loads(prompt.split("Input specs:\n", 1)[1])
        if self.first is None:
            self.first = request["items"][0]
        first = self.first
        return {
            "data": {
                "items": [
                    {
                        "id": item["id"],
                        "prompt": first["prompt"],
                        "chosen": first["reference_chosen"],
                        "rejected": first["reference_rejected"],
                        "metadata": first["metadata"],
                    }
                    for item in request["items"]
                ]
            },
            "telemetry": {},
        }


def test_duplicates_remain_non_counting_across_bounded_backfill(tmp_path):
    with pytest.raises(UnderfilledRunError):
        generate_llm_run(
            families=[FAMILY],
            count_per_family=2,
            batch_size=2,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="fake",
            generation_run="duplicate-backfill",
            max_tokens=64,
            concurrency=1,
            max_backfill_rounds=1,
            backend=_RepeatingBackend(),
        )

    manifest_path = tmp_path / "manifests" / f"{FAMILY}.duplicate-backfill.manifest.json"
    acceptance = json.loads(manifest_path.read_text())["metadata"]["dataset_acceptance"]
    assert acceptance["attempted_pairs"] == 3
    assert acceptance["accepted_pairs"] == 1
    assert acceptance["rejected_pairs"] == 2
    assert acceptance["duplicate_prompt_pairs"] == 2
    assert acceptance["duplicate_triple_pairs"] == 2
    assert acceptance["remaining_pairs"] == 1
