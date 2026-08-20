from slm_synth.sft.acceptance import (
    build_sft_content_summary,
    partition_unique_sft_rows,
    sft_conversation_fingerprint,
    sft_prompt_fingerprint,
)


def _row(row_id: str, prompt: str, response: str = "4") -> dict[str, object]:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "task_family": "grounded_qa_and_reading",
            "interaction_modes": ["single_turn"],
            "output_mode": "free_text",
            "context_mode": "supplied_passage",
            "difficulty": 1,
            "template_family": "direct_qa",
        },
    }


def test_sft_fingerprints_normalize_case_and_whitespace():
    left = _row("left", "  What is 2 + 2?\n", " FOUR ")
    right = _row("right", "what IS 2 + 2?", "four")

    assert sft_prompt_fingerprint(left) == sft_prompt_fingerprint(right)
    assert sft_conversation_fingerprint(left) == sft_conversation_fingerprint(right)


def test_partition_unique_sft_rows_preserves_first_and_reports_overlapping_reasons():
    rows = [
        _row("first", "What is 2 + 2?"),
        _row("second", " what IS 2 + 2? "),
        _row("third", "What is 3 + 1?"),
    ]

    accepted, summary = partition_unique_sft_rows(rows)

    assert [row["id"] for row in accepted] == ["first", "third"]
    assert summary == {
        "attempted_rows": 3,
        "accepted_rows": 2,
        "duplicate_rows": 1,
        "duplicate_reason_counts": {
            "duplicate_conversation": 1,
            "duplicate_prompt": 1,
        },
    }


def test_sft_content_summary_reports_response_repetition_without_rejecting_it():
    summary = build_sft_content_summary(
        [
            _row("first", "What is 2 + 2?", "4"),
            _row("second", "What is 3 + 1?", "4"),
        ]
    )

    assert summary["prompts"]["unique"] == 2
    assert summary["conversations"]["unique"] == 2
    assert summary["responses"]["unique"] == 1
    assert summary["responses"]["duplicate_count"] == 1
    assert summary["responses"]["maximum_repetition"] == 2
