from slm_synth.dpo.acceptance import (
    build_dpo_content_summary,
    dpo_prompt_fingerprint,
    dpo_triple_fingerprint,
    partition_unique_dpo_rows,
)


def _row(row_id: str, prompt: str, chosen: str = "4", rejected: str = "5") -> dict[str, object]:
    return {
        "id": row_id,
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "metadata": {
            "category": "answer_only_compliance",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": "basic_arithmetic_qa",
            "failure_mode": "wrong_numeric_answer",
        },
    }


def test_dpo_fingerprints_normalize_unicode_case_and_whitespace():
    left = _row("left", "  WHAT is ２ + ２?\n")
    right = _row("right", "what is 2 + 2?")

    assert dpo_prompt_fingerprint(left) == dpo_prompt_fingerprint(right)
    assert dpo_triple_fingerprint(left) == dpo_triple_fingerprint(right)


def test_partition_unique_dpo_rows_preserves_first_and_reports_reasons():
    accepted, summary = partition_unique_dpo_rows(
        [_row("first", "What is 2 + 2?"), _row("second", " what IS 2 + 2? ")]
    )

    assert [row["id"] for row in accepted] == ["first"]
    assert summary == {
        "attempted_pairs": 2,
        "accepted_pairs": 1,
        "duplicate_pairs": 1,
        "duplicate_reason_counts": {"duplicate_prompt": 1, "duplicate_triple": 1},
    }


def test_dpo_content_summary_reports_similarity_and_negative_patterns():
    summary = build_dpo_content_summary(
        [
            _row("one", "What is 2 + 2?", "4", "The answer is 4."),
            _row("two", "What is 3 + 1?", "4", "5"),
        ]
    )

    assert summary["chosen_rejected_similarity"]["maximum"] > 0
    assert summary["negative_patterns"]["counts"] == {
        "chosen_verbatim_with_extra": 1,
        "numeric_substitution": 1,
    }
    assert summary["chosen_responses"]["duplicate_count"] == 1
