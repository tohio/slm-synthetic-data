import json

import pytest

from slm_synth.sft.io import write_jsonl
from slm_synth.sft.report import build_coverage_report, require_publish_ready_report, write_coverage_report
from slm_synth.taxonomy.holdouts import HoldoutRecord, HoldoutRegistry


def _sft_row(
    *,
    row_id: str,
    category: str,
    eval_family: str,
    template_family: str,
    prompt: str | None = None,
    response: str | None = None,
) -> dict[str, object]:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": prompt or f"Answer item {row_id} with only the final value."},
            {"role": "assistant", "content": response or f"Result for {row_id}."},
        ],
        "metadata": {
            "task_family": eval_family,
            "interaction_modes": ["single_turn"],
            "output_mode": "free_text",
            "context_mode": "self_contained",
            "difficulty": 1,
            "template_family": template_family,
        },
    }


def test_build_sft_coverage_report_counts_metadata_across_files(tmp_path):
    arithmetic_path = tmp_path / "arithmetic.jsonl"
    repeat_path = tmp_path / "repeat.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="sft-arithmetic-1",
                category="answer_only_compliance",
                eval_family="grounded_qa_and_reading",
                template_family="worked_calculation",
                prompt="Calculate the remaining inventory after twelve of thirty crates ship.",
                response="Eighteen crates remain.",
            ),
            _sft_row(
                row_id="sft-arithmetic-2",
                category="answer_only_compliance",
                eval_family="grounded_qa_and_reading",
                template_family="direct_qa",
                prompt="According to the supplied biography, name the architect of the east wing.",
                response="Mara Okafor designed the east wing.",
            ),
        ],
        arithmetic_path,
    )
    write_jsonl(
        [
            _sft_row(
                row_id="sft-repeat-1",
                category="exact_output_format_control",
                eval_family="rewriting_and_editing",
                template_family="repeat_word_count",
            )
        ],
        repeat_path,
    )

    report = build_coverage_report([tmp_path])

    assert report["dataset_type"] == "sft"
    assert report["row_count"] == 3
    assert report["files"] == {
        str(arithmetic_path): 2,
        str(repeat_path): 1,
    }
    assert report["task_families"] == {
        "grounded_qa_and_reading": 2,
        "rewriting_and_editing": 1,
    }
    assert report["template_families"] == {
        "direct_qa": 1,
        "worked_calculation": 1,
        "repeat_word_count": 1,
    }
    assert report["difficulty_counts"] == {"1": 3}
    assert report["families"]["grounded_qa_and_reading"]["acceptance"] == {
        "attempted_rows": 2,
        "accepted_rows": 2,
        "estimated_tokens": 90,
        "rejected_rows": 0,
        "rejection_diagnostics": [],
        "duplicate_rows": 0,
        "candidate_rows": 2,
        "publish_ready": False,
        "publish_blockers": ["holdouts_not_checked"],
    }


def test_write_sft_coverage_report_writes_json(tmp_path):
    report_path = tmp_path / "reports" / "sft_coverage.json"
    report = {
        "dataset_type": "sft",
        "row_count": 0,
        "files": {},
        "task_families": {},
        "template_families": {},
        "difficulty_counts": {},
    }

    written = write_coverage_report(report=report, path=report_path)

    assert written == report_path
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_build_sft_coverage_report_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="input path does not exist"):
        build_coverage_report([tmp_path / "missing.jsonl"])


def test_build_sft_coverage_report_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no JSONL dataset files found"):
        build_coverage_report([tmp_path])


def test_sft_report_blocks_normalized_prompt_and_conversation_duplicates(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="first",
                category="answer_only_compliance",
                eval_family="grounded_qa_and_reading",
                template_family="direct_qa",
                prompt="What is 2 + 2?",
                response="4",
            ),
            _sft_row(
                row_id="second",
                category="answer_only_compliance",
                eval_family="grounded_qa_and_reading",
                template_family="direct_qa",
                prompt="  what IS 2 + 2? ",
                response="4",
            ),
        ],
        dataset_path,
    )

    report = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
    )

    assert report["content_uniqueness"]["prompts"]["duplicate_count"] == 1
    assert report["content_uniqueness"]["conversations"]["duplicate_count"] == 1
    assert report["acceptance"]["duplicate_rows"] == 1
    assert report["acceptance"]["candidate_rows"] == 2
    assert report["acceptance"]["publish_ready"] is False
    assert "duplicate_prompts" in report["acceptance"]["publish_blockers"]


def test_sft_report_blocks_repeated_assistant_response_clusters(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="first",
                category="answer_only_compliance",
                eval_family="grounded_qa_and_reading",
                template_family="direct_qa",
                prompt="What is 2 + 2?",
                response="4",
            ),
            _sft_row(
                row_id="second",
                category="answer_only_compliance",
                eval_family="grounded_qa_and_reading",
                template_family="comparison_qa",
                prompt="A ledger contains four signed receipts. State the receipt count.",
                response="4",
            ),
        ],
        dataset_path,
    )
    clean_registry = HoldoutRegistry([])

    report = build_coverage_report([dataset_path], holdout_registry=clean_registry)

    assert report["holdouts"] == {
        "status": "checked",
        "collision_count": 0,
        "collision_ids": [],
    }
    assert report["content_uniqueness"]["responses"]["duplicate_count"] == 1
    assert report["assistant_response_clusters"]["cluster_count"] == 1
    assert report["acceptance"]["publish_ready"] is False
    assert "repeated_assistant_response_clusters" in report["acceptance"]["publish_blockers"]
    with pytest.raises(ValueError, match="repeated_assistant_response_clusters"):
        require_publish_ready_report(report)

    collision_registry = HoldoutRegistry(
        [
            HoldoutRecord(
                id="holdout-1",
                eval_family="grounded_qa_and_reading",
                prompt="what is 2 + 2?",
                answer="4",
            )
        ]
    )
    collision_report = build_coverage_report(
        [dataset_path],
        holdout_registry=collision_registry,
    )
    assert collision_report["holdouts"]["collision_ids"] == ["first"]
    assert "holdout_collisions" in collision_report["acceptance"]["publish_blockers"]


def test_sft_report_marks_missing_holdout_check_as_publish_blocker(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="first",
                category="answer_only_compliance",
                eval_family="grounded_qa_and_reading",
                template_family="direct_qa",
            )
        ],
        dataset_path,
    )

    report = build_coverage_report([dataset_path])

    assert report["holdouts"]["status"] == "not_checked"
    assert report["acceptance"]["publish_blockers"] == ["holdouts_not_checked"]
    with pytest.raises(ValueError, match="holdouts_not_checked"):
        require_publish_ready_report(report)


def test_sft_report_records_candidate_and_rejection_outcomes_from_manifest(tmp_path):
    dataset_path = tmp_path / "arithmetic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="first",
                category="answer_only_compliance",
                eval_family="grounded_qa_and_reading",
                template_family="direct_qa",
            )
        ],
        dataset_path,
    )
    batch_manifest_path = tmp_path / "batch.manifest.json"
    batch_manifest_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "quality_adjudication": {
                        "first": {
                            "accepted": True,
                            "scores": {
                                "correctness": 4,
                                "grounding": 4,
                                "instruction_adherence": 4,
                                "completeness": 4,
                                "coherence": 4,
                            },
                            "constraint_results": [],
                        }
                    },
                    "deterministic_output_validation": {
                        "first": {
                            "status": "passed",
                            "declared_constraint_count": 0,
                            "checks": [],
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "run.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [{"batch_manifests": [str(batch_manifest_path)]}],
                "metadata": {
                    "publish_ready": True,
                    "candidate_rows": 2,
                    "rejection_reason_counts": {"batch_acceptance_error": 1},
                    "attempted_rows": 2,
                    "accepted_rows": 1,
                    "rejected_rows": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
        run_manifest=manifest_path,
    )

    assert report["acceptance"]["accepted_rows"] == 1
    assert report["acceptance"]["rejection_reason_counts"] == {"batch_acceptance_error": 1}
    assert report["acceptance"]["candidate_rows"] == 2
    assert report["acceptance"]["rejected_rows"] == 1
    assert report["acceptance"]["publish_ready"] is True


def test_sft_report_blocks_near_duplicate_prompts_and_conversations(tmp_path):
    dataset_path = tmp_path / "near.jsonl"
    common = "Summarize the supplied maintenance memo in exactly two concise sentences"
    write_jsonl(
        [
            _sft_row(
                row_id="near-1",
                category="unused",
                eval_family="summarization",
                template_family="memo_summary",
                prompt=f"{common} for the north facility.",
                response="The facility passed inspection; replace the north intake filter next week.",
            ),
            _sft_row(
                row_id="near-2",
                category="unused",
                eval_family="summarization",
                template_family="incident_summary",
                prompt=f"{common} for the south facility.",
                response="The facility passed inspection; replace the south intake filter next week.",
            ),
        ],
        dataset_path,
    )

    report = build_coverage_report([dataset_path], holdout_registry=HoldoutRegistry([]))

    assert report["near_duplicates"]["prompts"]["pair_count"] == 1
    assert report["near_duplicates"]["conversations"]["pair_count"] == 1
    assert "near_duplicate_prompts" in report["acceptance"]["publish_blockers"]
    assert "near_duplicate_conversations" in report["acceptance"]["publish_blockers"]


def test_sft_report_blocks_template_concentration(tmp_path):
    dataset_path = tmp_path / "templates.jsonl"
    rows = [
        _sft_row(
            row_id=f"template-{index}",
            category="unused",
            eval_family="classification_and_extraction",
            template_family="dominant" if index < 3 else "minority",
            prompt=f"Classify document {index} using its distinct policy and evidence record.",
        )
        for index in range(5)
    ]
    write_jsonl(rows, dataset_path)

    report = build_coverage_report([dataset_path], holdout_registry=HoldoutRegistry([]))

    assert report["template_concentration"]["concentrated_templates"] == [
        {"template_family": "dominant", "count": 3, "share": 0.6}
    ]
    assert "template_concentration" in report["acceptance"]["publish_blockers"]


def test_sft_report_blocks_invalid_tool_or_role_sequences(tmp_path):
    dataset_path = tmp_path / "invalid.jsonl"
    row = _sft_row(
        row_id="invalid-role",
        category="unused",
        eval_family="everyday_conversation",
        template_family="conversation",
    )
    row["messages"] = [
        {"role": "user", "content": "First request"},
        {"role": "user", "content": "Second adjacent request"},
        {"role": "assistant", "content": "Response"},
    ]
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = build_coverage_report([dataset_path], holdout_registry=HoldoutRegistry([]))

    assert report["row_count"] == 1
    assert report["valid_row_count"] == 0
    assert report["validation"]["invalid_tool_or_role_sequence_count"] == 1
    assert "invalid_public_rows" in report["acceptance"]["publish_blockers"]
    assert "invalid_tool_or_role_sequences" in report["acceptance"]["publish_blockers"]


def test_sft_report_requires_passing_semantic_evidence_for_every_public_row(tmp_path):
    dataset_path = tmp_path / "semantic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="semantic-pass",
                category="unused",
                eval_family="grounded_qa_and_reading",
                template_family="evidence_answer",
            ),
            _sft_row(
                row_id="semantic-fail",
                category="unused",
                eval_family="rewriting_and_editing",
                template_family="copy_edit",
            ),
        ],
        dataset_path,
    )
    batch_manifest_path = tmp_path / "batch.manifest.json"
    decisions = {}
    for row_id, accepted in (("semantic-pass", True), ("semantic-fail", False)):
        decisions[row_id] = {
            "accepted": accepted,
            "scores": {
                "correctness": 4 if accepted else 2,
                "grounding": 4,
                "instruction_adherence": 4,
                "completeness": 4,
                "coherence": 4,
            },
            "constraint_results": [],
        }
    batch_manifest_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "quality_adjudication": decisions,
                    "deterministic_output_validation": {
                        row_id: {
                            "status": "passed",
                            "declared_constraint_count": 0,
                            "checks": [],
                        }
                        for row_id in decisions
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    run_manifest_path = tmp_path / "run.manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "datasets": [{"batch_manifests": [str(batch_manifest_path)]}],
                "metadata": {"publish_ready": True},
            }
        ),
        encoding="utf-8",
    )

    report = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
        run_manifest=run_manifest_path,
    )

    assert report["semantic_adjudication"]["failed_row_ids"] == ["semantic-fail"]
    assert report["semantic_adjudication"]["status"] == "failed"
    assert "semantic_adjudication_failed" in report["acceptance"]["publish_blockers"]
    assert "semantic_adjudication_missing" not in report["acceptance"]["publish_blockers"]


def test_sft_report_blocks_missing_semantic_evidence_for_public_rows(tmp_path):
    dataset_path = tmp_path / "semantic.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="semantic-missing",
                category="unused",
                eval_family="grounded_qa_and_reading",
                template_family="evidence_answer",
            )
        ],
        dataset_path,
    )
    run_manifest_path = tmp_path / "run.manifest.json"
    run_manifest_path.write_text(
        json.dumps({"datasets": [], "metadata": {"publish_ready": True}}),
        encoding="utf-8",
    )

    report = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
        run_manifest=run_manifest_path,
    )

    assert report["semantic_adjudication"]["missing_row_ids"] == ["semantic-missing"]
    assert report["semantic_adjudication"]["status"] == "incomplete"
    assert "semantic_adjudication_missing" in report["acceptance"]["publish_blockers"]
    assert "deterministic_output_validation_missing" in report["acceptance"]["publish_blockers"]


def test_sft_report_blocks_failed_deterministic_output_evidence(tmp_path):
    dataset_path = tmp_path / "creative.jsonl"
    write_jsonl(
        [
            _sft_row(
                row_id="creative-fail",
                category="unused",
                eval_family="creative_writing",
                template_family="subtext_scene",
            )
        ],
        dataset_path,
    )
    batch_manifest_path = tmp_path / "batch.manifest.json"
    batch_manifest_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "quality_adjudication": {
                        "creative-fail": {
                            "accepted": True,
                            "scores": {
                                "correctness": 4,
                                "grounding": 4,
                                "instruction_adherence": 4,
                                "completeness": 4,
                                "coherence": 4,
                            },
                            "constraint_results": [],
                        }
                    },
                    "deterministic_output_validation": {
                        "creative-fail": {
                            "status": "failed",
                            "declared_constraint_count": 2,
                            "checks": [
                                {
                                    "constraint": "min_words",
                                    "expected": 500,
                                    "observed": 287,
                                    "passed": False,
                                }
                            ],
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    run_manifest_path = tmp_path / "run.manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "datasets": [{"batch_manifests": [str(batch_manifest_path)]}],
                "metadata": {"publish_ready": True},
            }
        ),
        encoding="utf-8",
    )

    report = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
        run_manifest=run_manifest_path,
    )

    assert report["deterministic_output_validation"]["failed_row_ids"] == [
        "creative-fail"
    ]
    assert "deterministic_output_constraint_failed" in report["acceptance"][
        "publish_blockers"
    ]
