from __future__ import annotations

from slm_synth.sft.schema import _normalize_adjacent_sft_role_messages


def _roles(row: dict) -> tuple[str, ...]:
    return tuple(message["role"] for message in row["messages"])


def test_adjacent_assistant_messages_are_merged() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Complete the function."},
            {"role": "assistant", "content": "line one"},
            {"role": "assistant", "content": "line two"},
        ]
    }

    normalized = _normalize_adjacent_sft_role_messages(row)

    assert _roles(normalized) == ("user", "assistant")
    assert normalized["messages"][1]["content"] == "line one\nline two"


def test_adjacent_user_messages_are_merged() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Part one."},
            {"role": "user", "content": "Part two."},
            {"role": "assistant", "content": "Answer."},
        ]
    }

    normalized = _normalize_adjacent_sft_role_messages(row)

    assert _roles(normalized) == ("user", "assistant")
    assert normalized["messages"][0]["content"] == "Part one.\nPart two."


def test_repeated_assistant_messages_are_merged() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Prompt."},
            {"role": "assistant", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ]
    }

    normalized = _normalize_adjacent_sft_role_messages(row)

    assert _roles(normalized) == ("user", "assistant")
    assert normalized["messages"][1]["content"] == "one\ntwo\nthree"


def test_system_user_user_assistant_normalizes_to_system_user_assistant() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Part one."},
            {"role": "user", "content": "Part two."},
            {"role": "assistant", "content": "Answer."},
        ]
    }

    normalized = _normalize_adjacent_sft_role_messages(row)

    assert _roles(normalized) == ("system", "user", "assistant")


def test_non_adjacent_multiturn_shape_is_not_collapsed() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Second."},
            {"role": "assistant", "content": "Second answer."},
        ]
    }

    normalized = _normalize_adjacent_sft_role_messages(row)

    assert _roles(normalized) == ("user", "assistant", "user", "assistant")
