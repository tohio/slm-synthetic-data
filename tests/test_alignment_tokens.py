import pytest

from slm_synth.runtime.reporting import estimate_dpo_tokens, estimate_sft_tokens


def test_alignment_token_estimates_ignore_ids_and_audit_metadata():
    sft = {"id": "one", "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}], "metadata": {"task_family": "conversation"}}
    changed = {**sft, "id": "a-much-longer-id", "metadata": {"task_family": "different", "notes": "x" * 100}}
    assert estimate_sft_tokens(sft) == estimate_sft_tokens(changed)


def test_dpo_estimate_counts_prompt_and_both_branches():
    row = {"prompt": [{"role": "user", "content": "Question"}], "chosen": [{"role": "assistant", "content": "Chosen answer"}], "rejected": [{"role": "assistant", "content": "Rejected answer"}]}
    assert estimate_dpo_tokens(row) > 0
    assert estimate_dpo_tokens({**row, "rejected": [{"role": "assistant", "content": "Rejected answer " * 20}]}) > estimate_dpo_tokens(row)


def test_alignment_token_estimates_reject_invalid_ratio():
    with pytest.raises(ValueError, match="chars_per_token"):
        estimate_sft_tokens({"messages": []}, chars_per_token=0)
