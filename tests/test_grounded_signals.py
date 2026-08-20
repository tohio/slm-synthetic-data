import json
from collections import Counter

import pytest

import slm_synth.pretrain.generate as generate
from slm_synth.pretrain.artifacts import (
    ArithmeticArtifactFactory,
    EducationalQAMCQGeneralArtifactFactory,
    EducationalQAMCQMathArtifactFactory,
    FactualRestraintArtifactFactory,
    TaskCodeArtifactFactory,
)
from slm_synth.pretrain.grounded import GroundedBatchStore, GroundedSignalGenerator
from slm_synth.llm import RetryableProviderExhaustedError, StructuredRenderedResponseError
from slm_synth.pretrain.artifacts.quality import (
    artifact_fingerprint,
    artifact_structure_fingerprint,
    validate_artifact,
)


class GroundedMockLLM:
    provider = "openrouter"
    model = "deepseek/deepseek-v4-flash"

    def generate_structured_object(self, *, prompt, schema, schema_name):
        signal = schema_name.removeprefix("grounded_").split("_batch_", 1)[0]
        payload = json.loads(prompt.split("GROUNDED ARTIFACTS:\n", 1)[1])
        records = []
        for item in payload:
            p = item["payload"]
            if signal == "arithmetic":
                nums = p["required_numeric_literals"]
                if item["family"] == "direct_expression":
                    question = f"What is the value of {p['expression']}?"
                else:
                    facts = "; ".join(p["facts"])
                    question = (
                        f"According to the {p['source']} for the {p['domain']}, "
                        f"the {p['item']} record states: {facts}. What quantity is requested?"
                    )
                records.append({"artifact_id": item["artifact_id"], "question": question, "steps": [f"The result is {p['answer']}."]})
            elif signal == "task_code":
                records.append({"artifact_id": item["artifact_id"], "plan": ["Process the inputs", "Return the result"]})
            elif signal == "educational_qa_mcq_math":
                records.append({"artifact_id": item["artifact_id"], "explanation": f"The verified calculation gives {p['answer']}."})
            elif signal == "educational_qa_mcq_general":
                records.append({"artifact_id": item["artifact_id"], "explanation": "The evidence directly supports the correct choice."})
            else:
                records.append({"artifact_id": item["artifact_id"], "safe_answer": "I can't determine or provide that from the supplied information; please use an appropriate reliable source or professional."})
        return {"records": records}


class GroundedInvalidFirstBatchLLM(GroundedMockLLM):
    def __init__(self):
        self.calls = 0

    def generate_structured_object(self, *, prompt, schema, schema_name):
        response = super().generate_structured_object(prompt=prompt, schema=schema, schema_name=schema_name)
        if self.calls == 0:
            response["records"][0]["artifact_id"] = response["records"][1]["artifact_id"]
        self.calls += 1
        return response


class GroundedRetryableProviderFirstBatchLLM(GroundedMockLLM):
    def __init__(self):
        self.calls = 0

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls += 1
        if self.calls == 1:
            raise RetryableProviderExhaustedError(
                "Retryable structured provider failure exhausted after 20 attempts: 429",
                telemetry={"retryable_provider_retries": 20},
            )
        return {
            "data": super().generate_structured_object(prompt=prompt, schema=schema, schema_name=schema_name),
            "telemetry": {},
        }


class GroundedMalformedFirstBatchLLM(GroundedMockLLM):
    def __init__(self):
        self.calls = 0

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls += 1
        if self.calls == 1:
            raise StructuredRenderedResponseError(
                "Structured rendered response unusable after 3 attempts: malformed JSON",
                telemetry={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}},
            )
        return {
            "data": super().generate_structured_object(prompt=prompt, schema=schema, schema_name=schema_name),
            "telemetry": {},
        }


class GroundedFailsLargeBatchesLLM(GroundedMockLLM):
    def __init__(self, maximum_successful_batch_size: int):
        self.maximum_successful_batch_size = maximum_successful_batch_size
        self.calls: list[int] = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        batch_size = int(schema_name.rsplit("_batch_", 1)[1])
        self.calls.append(batch_size)
        if batch_size > self.maximum_successful_batch_size:
            raise StructuredRenderedResponseError("batch too large", telemetry={})
        return {
            "data": super().generate_structured_object(prompt=prompt, schema=schema, schema_name=schema_name),
            "telemetry": {},
        }


@pytest.mark.parametrize("factory", [
    ArithmeticArtifactFactory,
    TaskCodeArtifactFactory,
    EducationalQAMCQMathArtifactFactory,
    EducationalQAMCQGeneralArtifactFactory,
    FactualRestraintArtifactFactory,
])
def test_artifact_factories_produce_distinct_batches(factory):
    capacity = getattr(factory, "UNIQUE_CANDIDATE_CAPACITY", 32)
    batch_size = min(32, capacity)
    rows = factory().build_batch(0, batch_size)
    assert len(rows) == batch_size
    assert len({row.artifact_id for row in rows}) == batch_size
    if factory is ArithmeticArtifactFactory:
        planned = [factory().build(index) for index in range(288)]
        assert len({row.family for row in planned}) == len(factory.FAMILIES)
        assert len({artifact_structure_fingerprint(row) for row in planned}) == len(planned)


def test_task_code_artifacts_are_valid_single_functions():
    import ast
    factory = TaskCodeArtifactFactory()
    for artifact in factory.build_batch(0, factory.UNIQUE_CANDIDATE_CAPACITY):
        tree = ast.parse(artifact.payload["code"])
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)


def test_task_code_catalog_has_no_renamed_structural_variants():
    factory = TaskCodeArtifactFactory()
    rows = [factory.build(index) for index in range(factory.UNIQUE_CANDIDATE_CAPACITY)]
    assert len(rows) == len(factory.FAMILIES) == 96
    assert len({artifact_structure_fingerprint(row) for row in rows}) == len(rows)
    with pytest.raises(ValueError, match="unique candidate capacity"):
        factory.build(factory.UNIQUE_CANDIDATE_CAPACITY)


def test_factual_restraint_catalog_has_distinct_scenarios_without_slot_variants():
    factory = FactualRestraintArtifactFactory()
    rows = [factory.build(index) for index in range(factory.UNIQUE_CANDIDATE_CAPACITY)]

    assert factory.UNIQUE_CANDIDATE_CAPACITY == len(factory.FAMILIES) == 32
    assert len({row.family for row in rows}) == len(rows)
    assert len({artifact_fingerprint(row) for row in rows}) == len(rows)
    assert len({artifact_structure_fingerprint(row) for row in rows}) == len(rows)
    assert all(not validate_artifact(row) for row in rows)

    with pytest.raises(ValueError, match="exceeds unique candidate capacity"):
        factory.build(factory.UNIQUE_CANDIDATE_CAPACITY)


def test_general_vocabulary_candidate_uses_an_adjective_compatible_subject():
    factory = EducationalQAMCQGeneralArtifactFactory()
    row = factory.build(factory.FAMILIES.index("vocabulary"))
    adjective = row.payload["question"].split()[2]
    answer, clue = factory.ADJECTIVE_CONTEXT[adjective]

    assert row.family == "vocabulary"
    assert row.payload["choices"][row.payload["correct_index"]] == answer
    assert clue in row.payload["evidence"]
    assert any(
        f"the {subject} as {adjective}" in row.payload["evidence"]
        for subject in factory.VOCABULARY_SUBJECTS[adjective]
    )


def test_general_mcq_catalog_covers_material_reasoning_families_once():
    factory = EducationalQAMCQGeneralArtifactFactory()
    rows = [factory.build(index) for index in range(factory.UNIQUE_CANDIDATE_CAPACITY)]

    assert len(rows) == len(factory.FAMILIES) == 24
    assert Counter(row.family for row in rows) == Counter({family: 1 for family in factory.FAMILIES})

    new_families = {
        "final_location", "table_lookup", "threshold_rule", "temporal_order",
        "direction_following", "conditional_access", "comparison_claim", "category_rule",
        "cause_inference", "schedule_availability", "inventory_shortage", "source_attribution",
        "procedure_step", "exception_rule", "trend_interpretation", "revision_tracking",
    }
    assert new_families.issubset(set(factory.FAMILIES))
    assert all(row.payload["choices"][row.payload["correct_index"]] == row.payload["answer"] for row in rows)
    assert all(len(set(row.payload["choices"])) == 4 for row in rows)

def test_general_mcq_choice_shuffle_is_deterministic_and_balanced():
    factory = EducationalQAMCQGeneralArtifactFactory()
    row_count = factory.UNIQUE_CANDIDATE_CAPACITY
    rows = [factory.build(index) for index in range(row_count)]
    replay = [factory.build(index) for index in range(row_count)]

    assert [row.payload["choices"] for row in rows] == [row.payload["choices"] for row in replay]
    assert [row.payload["correct_index"] for row in rows] == [row.payload["correct_index"] for row in replay]
    assert all(row.payload["choices"][row.payload["correct_index"]] == row.payload["answer"] for row in rows)

    distribution = Counter(row.payload["correct_index"] for row in rows)
    assert distribution == Counter({0: 6, 1: 6, 2: 6, 3: 6})


def test_all_grounded_generators_render_complete_batches():
    for signal in ("arithmetic", "task_code", "educational_qa_mcq_math", "educational_qa_mcq_general", "factual_restraint"):
        batch_size = 24 if signal in {"task_code", "educational_qa_mcq_math", "educational_qa_mcq_general"} else 32
        artifacts, records, telemetry = GroundedSignalGenerator(signal, GroundedMockLLM(), batch_size=batch_size).generate_batch(0)
        assert len(artifacts) == len(records) == batch_size
        assert all(record["type"] == signal for record in records)
        if signal == "educational_qa_mcq_general":
            assert all(record["evidence"] for record in records)
        if signal == "task_code":
            assert all("def " in record["code"] for record in records)


def test_batch_store_materializes_without_duplicates(tmp_path):
    artifacts, records, telemetry = GroundedSignalGenerator("factual_restraint", GroundedMockLLM(), batch_size=32).generate_batch(0)
    store = GroundedBatchStore(tmp_path, "factual_restraint")
    store.write_completed(batch_id=0, artifacts=artifacts, records=records)
    assert store.materialize_raw() == 32
    assert store.materialize_raw() == 32
    assert len(store.raw_path.read_text().splitlines()) == 32


def test_record_count_target_rounds_up_without_tokenizer():
    cfg = {"target_total_tokens": 5000, "generation": {"avg_tokens_per_sample": 80}}
    token_target, target_rows, rounded_rows = generate._rounded_batch_target_rows(
        cfg, {"target_tokens": 5000, "avg_tokens_per_sample": 60}, 32
    )
    assert token_target == 5000
    assert target_rows == 84
    assert rounded_rows == 96

    _, capped_rows, capped_rounded_rows = generate._rounded_batch_target_rows(
        cfg, {"target_tokens": 5000, "avg_tokens_per_sample": 60, "max_unique_candidates": 64}, 32
    )
    assert capped_rows == capped_rounded_rows == 64


def test_run_signal_resumes_from_completed_grounded_batches(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 32, "target_tokens": 90, "avg_tokens_per_sample": 90}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    generate.run_signal("factual_restraint", cfg, tmp_path)
    assert len((tmp_path / "raw" / "factual_restraint.jsonl").read_text().splitlines()) == 32
    generate.run_signal("factual_restraint", cfg, tmp_path)
    assert len((tmp_path / "raw" / "factual_restraint.jsonl").read_text().splitlines()) == 32


def test_run_signal_requeues_exhausted_retryable_provider_failure_and_continues(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v4-flash",
            "retries": {"exhausted_retryable_requeue_delay_seconds": 0},
        },
        "generation": {"batch_size": 32, "parallel_requests": 1},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 32, "samples": 32}},
    }
    llm = GroundedRetryableProviderFirstBatchLLM()
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: llm)

    generate.run_signal("factual_restraint", cfg, tmp_path)

    assert llm.calls > 1
    assert len((tmp_path / "raw" / "factual_restraint.jsonl").read_text().splitlines()) == 32
    rejection = (tmp_path / "rejected" / "factual_restraint.jsonl").read_text()
    assert "requeued_retryable_provider_failure" in rejection

    metrics = GroundedBatchStore(tmp_path, "factual_restraint").telemetry_summary()
    assert metrics["adaptive_batch_size_failures"] == 1
    assert metrics["adaptive_batch_size_observed_minimum"] == 2


def test_run_signal_retries_transient_rendered_batch_failure_at_smaller_size(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 1},
        "mix": {"arithmetic": {"architecture": "grounded", "batch_size": 32, "samples": 64}},
    }
    llm = GroundedInvalidFirstBatchLLM()
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: llm)

    generate.run_signal("arithmetic", cfg, tmp_path)

    raw_rows = (tmp_path / "raw" / "arithmetic.jsonl").read_text().splitlines()
    assert len(raw_rows) == 64
    rejection = (tmp_path / "rejected" / "arithmetic.jsonl").read_text()
    assert "adaptive_batch_size_reduced" in rejection

    metrics = GroundedBatchStore(tmp_path, "arithmetic").telemetry_summary()
    assert metrics["batches"] > 4
    assert metrics["dropped_batches"] == 0
    assert metrics["dropped_rows"] == 0
    assert metrics["adaptive_batch_size_failures"] == 1
    assert metrics["adaptive_batch_size_decreases"] == 1
    assert metrics["adaptive_batch_size_observed_minimum"] == 2

    generate.run_signal("arithmetic", cfg, tmp_path)
    assert len((tmp_path / "raw" / "arithmetic.jsonl").read_text().splitlines()) == 64


def test_run_signal_reduces_grounded_batch_size_after_rendered_failure(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 8, "parallel_requests": 1},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 8, "samples": 8}},
    }
    llm = GroundedFailsLargeBatchesLLM(maximum_successful_batch_size=2)
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: llm)

    generate.run_signal("factual_restraint", cfg, tmp_path)

    assert llm.calls == [4, 2, 2, 2, 2]
    assert len((tmp_path / "raw" / "factual_restraint.jsonl").read_text().splitlines()) == 8
    metrics = GroundedBatchStore(tmp_path, "factual_restraint").telemetry_summary()
    assert metrics["adaptive_batch_size_observed_minimum"] == 2
    assert metrics["adaptive_batch_size_decreases"] == 1


@pytest.mark.parametrize("signal", [
    "arithmetic",
    "task_code",
    "educational_qa_mcq_math",
    "educational_qa_mcq_general",
    "factual_restraint",
])
def test_run_signal_retries_exhausted_malformed_structured_response_for_every_grounded_signal(monkeypatch, tmp_path, signal):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 1},
        "mix": {signal: {"architecture": "grounded", "batch_size": 32, "samples": 64, **({"max_unique_candidates": 24} if signal in {"task_code", "educational_qa_mcq_math", "educational_qa_mcq_general"} else {"max_unique_candidates": 32} if signal == "factual_restraint" else {})}},
    }
    llm = GroundedMalformedFirstBatchLLM()
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: llm)

    generate.run_signal(signal, cfg, tmp_path)

    expected_rows = 24 if signal in {"task_code", "educational_qa_mcq_math", "educational_qa_mcq_general"} else 32 if signal == "factual_restraint" else 64
    assert len((tmp_path / "raw" / f"{signal}.jsonl").read_text().splitlines()) == expected_rows
    rejection = (tmp_path / "rejected" / f"{signal}.jsonl").read_text()
    assert "adaptive_batch_size_reduced" in rejection

    metrics = GroundedBatchStore(tmp_path, signal).telemetry_summary()
    assert metrics["batches"] > 4
    assert metrics["dropped_batches"] == 0
    assert metrics["dropped_rows"] == 0
    assert metrics["adaptive_batch_size_failures"] == 1
    assert metrics["adaptive_batch_size_decreases"] == 1
    assert metrics["adaptive_batch_size_observed_minimum"] == 2
    assert metrics["cost"] == 0.0


def test_grounded_artifacts_have_no_placeholder_quality_failures():
    for factory in (EducationalQAMCQMathArtifactFactory, EducationalQAMCQGeneralArtifactFactory, FactualRestraintArtifactFactory):
        capacity = getattr(factory, "UNIQUE_CANDIDATE_CAPACITY", 512)
        rows = [factory().build(index) for index in range(min(512, capacity))]
        assert all(not validate_artifact(row) for row in rows)
        assert len({artifact_fingerprint(row) for row in rows}) == len(rows)


def test_general_revision_tracking_does_not_duplicate_note_noun():
    artifact = EducationalQAMCQGeneralArtifactFactory().build(23)
    public_source = artifact.payload["evidence"] + " " + artifact.payload["question"]
    assert "note note" not in public_source


def test_math_mcq_positive_quantity_families_have_nonnegative_plausible_choices():
    factory = EducationalQAMCQMathArtifactFactory()
    for index in range(factory.UNIQUE_CANDIDATE_CAPACITY):
        artifact = factory.build(index)
        assert all(int(choice) >= 0 for choice in artifact.payload["choices"])


def test_general_mcq_has_one_structurally_distinct_candidate_per_family():
    factory = EducationalQAMCQGeneralArtifactFactory()
    artifacts = [factory.build(index) for index in range(factory.UNIQUE_CANDIDATE_CAPACITY)]

    assert factory.UNIQUE_CANDIDATE_CAPACITY == len(factory.FAMILIES) == 24
    assert len({artifact.family for artifact in artifacts}) == len(artifacts)
    assert len({artifact_structure_fingerprint(artifact) for artifact in artifacts}) == len(artifacts)

    renderer = GroundedSignalGenerator(
        "educational_qa_mcq_general",
        GroundedMockLLM(),
        batch_size=len(artifacts),
        factory=factory,
    )
    assert renderer.response_schema()["properties"]["records"]["items"]["required"] == [
        "artifact_id",
        "explanation",
    ]

    with pytest.raises(ValueError, match="exceeds unique candidate capacity"):
        factory.build(factory.UNIQUE_CANDIDATE_CAPACITY)


def test_batch_store_persists_telemetry(tmp_path):
    artifacts, records, _ = GroundedSignalGenerator("factual_restraint", GroundedMockLLM(), batch_size=32).generate_batch(0)
    store = GroundedBatchStore(tmp_path, "factual_restraint")
    store.write_completed(
        batch_id=0, artifacts=artifacts, records=records,
        telemetry={"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.01}, "elapsed_seconds": 1.5, "retry_count": 1, "retryable_provider_retries": 1, "retry_sleep_seconds": 4.5, "adaptive_window_increases": 2, "adaptive_window_decreases": 1, "adaptive_admission_wait_seconds": 5.0, "adaptive_peak_in_flight_limit": 256, "adaptive_min_in_flight_limit": 64, "max_adaptive_cooldown_seconds": 5.0},
    )
    assert store.telemetry_summary()["total_tokens"] == 30
    assert store.telemetry_summary()["cost"] == 0.01
    assert store.telemetry_summary()["retryable_provider_retries"] == 1
    assert store.telemetry_summary()["retry_sleep_seconds"] == 4.5
    assert store.telemetry_summary()["adaptive_window_increases"] == 2
    assert store.telemetry_summary()["adaptive_window_decreases"] == 1
    assert store.telemetry_summary()["adaptive_admission_wait_seconds"] == 5.0
    assert store.telemetry_summary()["adaptive_peak_in_flight_limit"] == 256
    assert store.telemetry_summary()["adaptive_min_in_flight_limit"] == 64
    assert store.telemetry_summary()["max_adaptive_cooldown_seconds"] == 5.0
    assert store.telemetry_summary()["aggregate_request_seconds"] == 1.5
    assert "elapsed_seconds" not in store.telemetry_summary()


def test_run_signal_materializes_raw_only_at_start_and_completion(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 1},
        "mix": {"arithmetic": {"architecture": "grounded", "batch_size": 32, "samples": 96}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    original = GroundedBatchStore.materialize_raw
    calls = []

    def counted(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(GroundedBatchStore, "materialize_raw", counted)
    generate.run_signal("arithmetic", cfg, tmp_path)

    assert len(calls) == 2
    assert len((tmp_path / "raw" / "arithmetic.jsonl").read_text().splitlines()) == 96


def test_run_signal_supports_bounded_concurrent_grounded_batches(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 2},
        "mix": {"arithmetic": {"architecture": "grounded", "batch_size": 32, "samples": 64}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    generate.run_signal("arithmetic", cfg, tmp_path)
    assert len((tmp_path / "raw" / "arithmetic.jsonl").read_text().splitlines()) == 64
    metrics = GroundedBatchStore(tmp_path, "arithmetic").telemetry_summary()
    batch_manifests = list((tmp_path / "manifests" / "grounded" / "arithmetic" / "batches").glob("batch_*.json"))
    assert len(batch_manifests) == metrics["batches"]
    assert metrics["batches"] > 2
    assert metrics["adaptive_batch_size_observed_minimum"] == 4
    assert metrics["adaptive_batch_size_observed_peak"] >= 8


def test_run_signal_supports_batch_size_64_for_qualification(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 64, "parallel_requests": 8},
        "mix": {"arithmetic": {"architecture": "grounded", "batch_size": 64, "samples": 64}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    generate.run_signal("arithmetic", cfg, tmp_path)
    assert len((tmp_path / "raw" / "arithmetic.jsonl").read_text().splitlines()) == 64


def test_run_signal_rejects_batch_size_above_qualification_limit(tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 65, "parallel_requests": 8},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 65, "samples": 65}},
    }
    with pytest.raises(ValueError, match="batch_size between 1 and 64"):
        generate.run_signal("factual_restraint", cfg, tmp_path)


def test_run_signal_supports_concurrency_1024_for_qualification(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 1024},
        "mix": {"arithmetic": {"architecture": "grounded", "batch_size": 32, "samples": 64}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    generate.run_signal("arithmetic", cfg, tmp_path)
    assert len((tmp_path / "raw" / "arithmetic.jsonl").read_text().splitlines()) == 64


def test_run_signal_rejects_concurrency_above_qualification_limit(tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 1025},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 32, "samples": 64}},
    }
    with pytest.raises(ValueError, match="parallel_requests between 1 and 1024"):
        generate.run_signal("factual_restraint", cfg, tmp_path)

def test_general_mcq_count_pattern_uses_deterministic_explanation():
    from slm_synth.pretrain.artifacts.educational_qa_mcq_general import EducationalQAMCQGeneralArtifactFactory
    from slm_synth.pretrain.grounded import GroundedSignalGenerator

    factory = EducationalQAMCQGeneralArtifactFactory()

    artifact = None
    for index in range(5000):
        candidate = factory.build(index)
        evidence = str(candidate.payload.get("evidence", ""))
        if "values.count(" in evidence:
            artifact = candidate
            break

    assert artifact is not None

    renderer = GroundedSignalGenerator(
        signal="educational_qa_mcq_general",
        factory=factory,
        llm=None,
        batch_size=1,
    )

    bad_llm_row = {
        "explanation": "Incorrectly, the count is three occurrences."
    }

    record = renderer._finalize(artifact, bad_llm_row)

    assert "three occurrences" not in record["explanation"].lower()
    assert "`values.count(" in record["explanation"]
    assert record["choices"][record["correct_index"]] in record["explanation"]
