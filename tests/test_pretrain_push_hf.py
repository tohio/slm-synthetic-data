import json

import pytest

from slm_synth.pretrain.push_hf import require_complete_accepted_token_report


def test_pretrain_push_requires_complete_accepted_token_report(tmp_path):
    path = tmp_path / "accepted_token_report.json"
    path.write_text(
        json.dumps({"status": "shortfall", "publish_ready": False, "token_deficit": 42}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has not reached"):
        require_complete_accepted_token_report(path)

    path.write_text(
        json.dumps({"status": "complete", "publish_ready": True, "token_deficit": 0}),
        encoding="utf-8",
    )
    assert require_complete_accepted_token_report(path)["status"] == "complete"
