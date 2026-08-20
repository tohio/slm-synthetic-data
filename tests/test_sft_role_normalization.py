import pytest

from slm_synth.sft.schema import validate_messages


def test_adjacent_roles_are_rejected_not_silently_merged():
    with pytest.raises(ValueError, match="malformed role sequence"):
        validate_messages([
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Answer"},
        ])
