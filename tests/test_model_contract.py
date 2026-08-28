import json
import pytest

from slm_synth.qualify_model import qualify

def test_model_qualification_uses_the_same_structured_contract(monkeypatch):
    class Backend:
        def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
            if "pretrain_judge" in schema_name:
                data = {
                    "decisions": {
                        "probe": {
                            "assessable": False,
                            "quality_valid": False,
                            "signal_aligned": False,
                            "natural_and_useful": False,
                            "reason": "insufficient evidence",
                        }
                    }
                }
            elif "judge" in schema_name:
                data = {
                    "decisions": {
                        "probe": {
                            "assessable": False,
                            "accepted": False,
                            "reason": "insufficient evidence",
                        }
                    }
                }
            elif "reviewer" in schema_name:
                data = {
                    "decisions": {
                        "probe": {
                            "agreed": False,
                            "reason": "unsupported claim",
                        }
                    }
                }
            else:
                data = {"items": [{"content": "Grounded answer."}]}
            return {"data": data, "telemetry": {"request_count": 1}}

    class Suitability:
        def as_dict(self):
            return {
                "model": "example/model",
                "reasoning_capable": False,
                "reasoning_mandatory": False,
                "reasoning_disable_supported": True,
                "reasoning_policy_pass": True,
                "source": "OpenRouter",
            }

    monkeypatch.setattr("slm_synth.qualify_model.get_reasoning_suitability", lambda model: Suitability())
    monkeypatch.setattr("slm_synth.qualify_model.build_backend", lambda **kwargs: Backend())
    report = qualify(
        model="example/model",
        roles=["sft-generator", "sft-judge", "sft-reviewer"],
        max_tokens=128,
        routing_mode="auto",
    )

    assert report["contract"] == "openrouter_strict_json_schema_v1"
    assert report["schema_version"] == 6
    assert report["passed"] is True
    assert all(result["passed"] for result in report["roles"].values())
