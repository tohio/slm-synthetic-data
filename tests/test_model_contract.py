import json
from argparse import Namespace

import pytest

from slm_synth.estimate_generation_cost import estimate
from slm_synth.llm import LLMBackend
from slm_synth.model_contract import (
    PlainOutputContractError,
    call_plain_parsed,
    parse_json_object,
    parse_judge_decision,
    parse_review_decision,
)
from slm_synth.qualify_model import qualify
from slm_synth.tool_catalog import tools_from_spec


def test_plain_json_parser_accepts_only_one_object():
    assert parse_json_object('```json\n{"items": []}\n```') == {"items": []}
    with pytest.raises(ValueError, match="one valid JSON object"):
        parse_json_object('prefix {"items": []}')


def test_plain_request_omits_provider_specific_and_sampling_parameters():
    backend = LLMBackend.__new__(LLMBackend)
    backend.model = "example/model"
    backend.max_tokens = 256
    backend.temperature = None
    backend.top_p = None
    backend._provider_extra_body = lambda: {"provider": {"allow_fallbacks": True}}
    kwargs = backend._plain_text_kwargs(system_prompt="system", prompt="user")
    assert set(kwargs) == {"model", "messages", "max_tokens", "extra_body"}
    assert "response_format" not in kwargs


def test_decision_parsers_are_small_and_conservative():
    judge = parse_judge_decision("ASSESSABLE: NO\nDECISION: ACCEPT\nREASON: missing evidence")
    assert judge.accepted is False
    review = parse_review_decision("AGREE: NO\nREASON: unsupported claim")
    assert review.agreed is False


def test_tools_are_code_owned_and_deterministic():
    spec = {
        "id": "tool-1",
        "metadata": {"interaction_modes": ["single_turn", "tool_mediated"]},
        "variables": {"tool": "forecast(city, country, date)"},
    }
    tools = tools_from_spec(spec)
    assert tools[0]["function"]["name"] == "forecast"
    assert set(tools[0]["function"]["parameters"]["required"]) == {"city", "country", "date"}


def test_cost_estimator_counts_reviewer_only_after_judge_acceptance(tmp_path):
    pricing = {
        "data": [
            {
                "id": "gen",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            },
            {
                "id": "judge",
                "pricing": {"prompt": "0.000001", "completion": "0.000001"},
            },
            {
                "id": "review",
                "pricing": {"prompt": "0.000001", "completion": "0.000001"},
            },
        ]
    }
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(pricing))
    args = Namespace(
        generator_model="gen",
        judge_model="judge",
        reviewer_model="review",
        candidates=100,
        target_accepted=None,
        target_tokens=None,
        average_accepted_tokens=500,
        pricing_file=str(path),
        deterministic_pass_rate=1.0,
        judge_acceptance=0.5,
        reviewer_agreement=0.8,
        generator_input_tokens=100,
        generator_output_tokens=50,
        judge_input_tokens=100,
        judge_output_tokens=10,
        reviewer_input_tokens=100,
        reviewer_output_tokens=10,
    )
    report = estimate(args)["scenarios"]["expected"]
    assert report["estimated_accepted"] == 40
    assert report["role_costs_usd"]["reviewer"] == pytest.approx(50 * 110 * 0.000001)


def test_model_qualification_uses_the_same_plain_contract(monkeypatch):
    class Backend:
        judge_attempts = 0

        def generate_text_with_metadata(self, *, prompt, system_prompt):
            if "ASSESSABLE" in prompt:
                self.judge_attempts += 1
                if self.judge_attempts == 1:
                    text = "ASSESSABLE: NO\nDECISION: REJECT"
                else:
                    text = "ASSESSABLE: NO\nDECISION: REJECT\nREASON: insufficient context"
            elif "AGREE" in prompt:
                text = "AGREE: NO\nREASON: the answer was unsupported"
            else:
                text = '{"items":[{"content":"Grounded answer."}]}'
            return {"text": text, "telemetry": {"request_count": 1}}

    monkeypatch.setattr("slm_synth.qualify_model.LLMBackend", lambda **kwargs: Backend())
    report = qualify(
        model="example/model",
        roles=["sft-generator", "sft-judge", "sft-reviewer"],
        max_tokens=128,
        routing_mode="auto",
    )

    assert report["contract"] == "portable_plain_text_v1"
    assert report["schema_version"] == 2
    assert report["passed"] is True
    assert all(result["passed"] for result in report["roles"].values())
    assert report["roles"]["sft-judge"]["transport_compatible"] is True
    assert report["roles"]["sft-judge"]["contract_pass"] is True
    assert report["roles"]["sft-judge"]["telemetry"]["batch_count"] == 2


def test_plain_parser_error_retains_transport_evidence_after_bounded_retries():
    class Backend:
        calls = 0

        def generate_text_with_metadata(self, *, prompt, system_prompt):
            self.calls += 1
            return {
                "text": "ASSESSABLE: NO\nDECISION: REJECT",
                "telemetry": {"usage": {"total_tokens": 10}},
            }

    backend = Backend()
    with pytest.raises(PlainOutputContractError) as error:
        call_plain_parsed(
            backend,
            prompt="judge this",
            system_prompt="judge",
            parser=parse_judge_decision,
            attempts=3,
        )

    assert backend.calls == 3
    assert error.value.response == "ASSESSABLE: NO\nDECISION: REJECT"
    assert error.value.telemetry["batch_count"] == 3
    assert error.value.telemetry["usage"]["total_tokens"] == 30
