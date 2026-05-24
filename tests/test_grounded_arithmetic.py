import json
from pathlib import Path

import pytest

import slm_synth.generate as generate
from slm_synth.artifacts.arithmetic import ArithmeticArtifactFactory
from slm_synth.grounded import CorpusTokenCounter, GroundedArithmeticGenerator, GroundedBatchStore


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class GroundedMockLLM:
    provider = "openrouter"
    model = "deepseek/deepseek-v4-flash"

    def generate_structured_object(self, *, prompt, schema, schema_name):
        assert schema_name == "grounded_arithmetic_batch_32"
        payload = json.loads(prompt.split("GROUNDED ARTIFACTS:\n", 1)[1])
        records = []
        for item in payload:
            p = item["payload"]
            if "expression" in p and item["family"] == "direct_expression":
                question = f"What is the value of {p['expression']}?"
            elif item["family"] == "missing_operand":
                a, total = p["required_numeric_literals"]
                question = f"After {a} items are added, the total is {total}. How many were there at first?"
            elif item["family"] == "two_step_remaining_quantity":
                start, first, second = p["required_numeric_literals"]
                question = f"Start with {start} tickets, remove {first}, then remove {second}. How many remain?"
            elif item["family"] == "exact_allocation":
                total, per = p["required_numeric_literals"]
                question = f"There are {total} items and each tray holds {per}. How many trays are needed?"
            else:
                numbers = p["required_numeric_literals"]
                question = f"Which result is largest: {numbers[0]} + {numbers[1]}, {numbers[2]} + {numbers[3]}, or {numbers[4]} + {numbers[5]}?"
            records.append({
                "artifact_id": item["artifact_id"],
                "question": question,
                "steps": [f"Compute the verified expression to get {p['answer']}."],
            })
        return {"records": records}


def test_arithmetic_factory_produces_distinct_verified_artifacts():
    rows = ArithmeticArtifactFactory().build_batch(0, 32)
    assert len(rows) == 32
    assert len({row.artifact_id for row in rows}) == 32
    assert {row.family for row in rows} == set(ArithmeticArtifactFactory.FAMILIES)
    assert all(row.payload["answer"] and row.payload["expression"] for row in rows)


def test_grounded_arithmetic_generator_anchors_answer_and_verification_locally():
    artifacts, records = GroundedArithmeticGenerator(GroundedMockLLM(), batch_size=32).generate_batch(0)
    assert len(artifacts) == len(records) == 32
    for artifact, record in zip(artifacts, records):
        assert record["answer"] == artifact.payload["answer"]
        assert record["verification_expression"] == artifact.payload["expression"]
        assert record["verification_answer"] == artifact.payload["answer"]


def test_grounded_arithmetic_generator_rejects_numeric_drift():
    class BadLLM(GroundedMockLLM):
        def generate_structured_object(self, **kwargs):
            output = super().generate_structured_object(**kwargs)
            output["records"][0]["question"] = "What is 999 + 1?"
            return output

    with pytest.raises(ValueError, match="changed numeric facts"):
        GroundedArithmeticGenerator(BadLLM(), batch_size=32).generate_batch(0)


def test_batch_store_materializes_raw_without_duplicate_rows(tmp_path):
    factory = ArithmeticArtifactFactory()
    artifacts = factory.build_batch(0, 2)
    records = [
        {"type": "arithmetic", "question": "What is 1 + 1?", "steps": ["1 + 1 = 2"], "answer": "2"},
        {"type": "arithmetic", "question": "What is 2 + 2?", "steps": ["2 + 2 = 4"], "answer": "4"},
    ]
    store = GroundedBatchStore(tmp_path, "arithmetic")
    store.write_completed(batch_id=0, artifacts=artifacts, records=records, token_count=10)
    assert store.total_tokens() == 10
    assert store.materialize_raw() == 2
    assert store.materialize_raw() == 2
    assert len(store.raw_path.read_text().splitlines()) == 2


def test_token_counter_counts_final_arithmetic_text():
    counter = CorpusTokenCounter(WhitespaceTokenizer())
    count = counter.count_arithmetic_records([
        {"question": "What is 2 + 2?", "steps": ["2 + 2 = 4"], "answer": "4"}
    ])
    assert count == len("What is 2 + 2?\n2 + 2 = 4\n4".split())


def test_run_signal_routes_grounded_arithmetic_and_resumes_from_completed_batch(monkeypatch, tmp_path):
    cfg = {
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "tokenizer_name_or_path": "fake"},
        "mix": {"arithmetic": {"architecture": "grounded", "batch_size": 32, "target_tokens": 1}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    monkeypatch.setattr(
        generate.CorpusTokenCounter,
        "from_pretrained",
        classmethod(lambda cls, _: cls(WhitespaceTokenizer())),
    )
    generate.run_signal("arithmetic", cfg, tmp_path)
    raw_rows = (tmp_path / "raw" / "arithmetic.jsonl").read_text().splitlines()
    assert len(raw_rows) == 32
    # Second invocation should see that target is already met and not append rows.
    generate.run_signal("arithmetic", cfg, tmp_path)
    assert len((tmp_path / "raw" / "arithmetic.jsonl").read_text().splitlines()) == 32


def test_openrouter_structured_request_uses_schema_and_provider_routing(monkeypatch):
    from slm_synth.llm import LLMBackend

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    backend = LLMBackend(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        max_tokens=2048,
        temperature=0.35,
        top_p=0.95,
    )
    captured = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return object()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    backend.client = Client()
    backend._create_structured_completion("prompt", {"type": "object"}, "schema_name")
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["extra_body"]["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": False,
    }


def test_transport_interruption_is_retryable():
    from slm_synth.llm import LLMBackend

    backend = LLMBackend.__new__(LLMBackend)
    assert backend._is_transient_transport_error(
        RuntimeError("peer closed connection without sending complete message body (incomplete chunked read)")
    )
