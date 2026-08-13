import json
from copy import deepcopy

from slm_synth.distillation_dpo.acceptance import build_response_pattern_report
from slm_synth.distillation_dpo.report import build_coverage_report
from slm_synth.distillation_dpo.spec_builders import build_production_rows
from slm_synth.taxonomy.holdouts import HoldoutRegistry


FAMILY = "teacher_response_preference"


def test_report_lists_full_rejected_response_clusters_without_blocking(tmp_path):
    source_rows = build_production_rows(family=FAMILY, count=40)
    rows = [
        deepcopy(row)
        for row in source_rows
        if row["metadata"]["category"] == "general_instruction_following"
    ][:3]
    repeated = "This response ignores the requested constraints and gives no actionable recommendation."
    for row in rows:
        row["rejected"] = [{"role": "assistant", "content": repeated}]

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
    )
    rejected = report["response_patterns"]["rejected_responses"]

    assert rejected["total"] == 3
    assert rejected["unique"] == 1
    assert rejected["duplicate_count"] == 2
    assert rejected["maximum_repetition"] == 3
    assert rejected["repeated_cluster_count"] == 1
    cluster = rejected["repeated_clusters"][0]
    assert cluster["response"] == repeated
    assert cluster["count"] == 3
    assert cluster["row_ids"] == [row["id"] for row in rows]
    assert cluster["prompts"] == [row["prompt"][-1]["content"] for row in rows]
    assert cluster["categories"] == ["general_instruction_following"]
    assert cluster["failure_modes"] == sorted(
        {row["metadata"]["failure_mode"] for row in rows}
    )
    assert report["dataset_acceptance"]["publish_blockers"] == []
    assert report["dataset_acceptance"]["publish_ready"] is True


def test_response_pattern_report_summarizes_similarity_and_negative_constructions():
    rows = [
        deepcopy(row)
        for row in build_production_rows(family=FAMILY, count=40)
        if row["metadata"]["category"] == "direct_arithmetic"
    ][:2]

    patterns = build_response_pattern_report(rows)

    similarity = patterns["chosen_rejected_similarity"]
    assert 0.0 <= similarity["minimum"] <= similarity["mean"] <= similarity["maximum"] <= 1.0
    assert patterns["negative_patterns"]["counts"] == {"numeric_substitution": 2}
    assert patterns["negative_patterns"]["row_ids"] == {
        "numeric_substitution": [row["id"] for row in rows]
    }
    assert patterns["negative_patterns"]["maximum_repetition"] == 2
    assert patterns["policy"] == {
        "repeated_response_clusters_are_diagnostic": True,
        "similarity_is_diagnostic": True,
        "negative_patterns_are_diagnostic": True,
        "automatic_semantic_judge": False,
    }


def test_similarity_report_lists_high_similarity_row_ids():
    row = deepcopy(build_production_rows(family=FAMILY, count=1)[0])
    row["chosen"] = [
        {"role": "assistant", "content": "Use three measured steps to scale the service safely."}
    ]
    row["rejected"] = [
        {"role": "assistant", "content": "Use three measured steps to scale the service safely!"}
    ]

    similarity = build_response_pattern_report([row])["chosen_rejected_similarity"]

    assert similarity["at_or_above_0_90"] == 1
    assert similarity["at_or_above_0_98"] == 1
    assert similarity["rows_at_or_above_0_90"][0]["id"] == row["id"]
