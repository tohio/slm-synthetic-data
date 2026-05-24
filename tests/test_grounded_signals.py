import json

import pytest

import slm_synth.generate as generate
from slm_synth.artifacts import (
    ArithmeticArtifactFactory,
    EducationalQAMCQGeneralArtifactFactory,
    EducationalQAMCQMathArtifactFactory,
    FactualRestraintArtifactFactory,
    TaskCodeArtifactFactory,
)
from slm_synth.grounded import GroundedBatchStore, GroundedSignalGenerator
from slm_synth.artifacts.quality import artifact_fingerprint, validate_artifact


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
                question = "Compute " + " + ".join(nums) + "."
                if item["family"] == "direct_expression":
                    question = f"What is the value of {p['expression']}?"
                elif item["family"] == "missing_operand":
                    question = f"After {nums[0]} items are added, the total is {nums[1]}. How many were there first?"
                elif item["family"] == "two_step_remaining_quantity":
                    question = f"Start with {nums[0]} items, remove {nums[1]}, then remove {nums[2]}. How many remain?"
                elif item["family"] == "exact_allocation":
                    question = f"There are {nums[0]} items packed {nums[1]} per box. How many boxes are needed?"
                elif item["family"] == "unique_numeric_comparison":
                    question = "Which has the largest value: " + ", ".join(p["expressions"]) + "?"
                records.append({"artifact_id": item["artifact_id"], "question": question, "steps": [f"The result is {p['answer']}."]})
            elif signal == "task_code":
                records.append({"artifact_id": item["artifact_id"], "task": "Write a Python function that implements the supplied behavior and returns a new result without mutating inputs.", "plan": ["Process the inputs", "Return the result"]})
            elif signal == "educational_qa_mcq_math":
                records.append({"artifact_id": item["artifact_id"], "question": "Find the value using " + " and ".join(p["required_numeric_literals"]) + ".", "explanation": f"The verified answer is {p['answer']}."})
            elif signal == "educational_qa_mcq_general":
                records.append({"artifact_id": item["artifact_id"], "explanation": "The evidence directly supports the held correct choice."})
            else:
                records.append({"artifact_id": item["artifact_id"], "safe_answer": "I can't determine or provide that from the supplied information; please use an appropriate reliable source or professional."})
        return {"records": records}


@pytest.mark.parametrize("factory", [
    ArithmeticArtifactFactory,
    TaskCodeArtifactFactory,
    EducationalQAMCQMathArtifactFactory,
    EducationalQAMCQGeneralArtifactFactory,
    FactualRestraintArtifactFactory,
])
def test_artifact_factories_produce_distinct_batches(factory):
    rows = factory().build_batch(0, 32)
    assert len(rows) == 32
    assert len({row.artifact_id for row in rows}) == 32


def test_task_code_artifacts_are_valid_single_functions():
    import ast
    for artifact in TaskCodeArtifactFactory().build_batch(0, 32):
        tree = ast.parse(artifact.payload["code"])
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)


def test_task_code_omits_zero_collision_variant_without_removing_semantic_zeroes():
    import ast

    factory = TaskCodeArtifactFactory()
    def function_name(payload):
        return ast.parse(payload["code"]).body[0].name

    assert function_name(factory._build_normalized_counting(0)) == "count_clean_tags_lower_1"
    assert function_name(factory._build_paired_comparison_counts(0)) == "compare_pairs_with_margin_0"
    assert function_name(factory._build_dictionary_keywise_sum(0)) == "combine_tag_counts_min_0"
    assert function_name(factory._build_paired_comparison_counts(101)) == "compare_pairs_with_margin_0_1"

    rows = [factory.build(index) for index in range(2016)]
    assert len({row.payload["code"] for row in rows}) == len(rows)


def test_factual_restraint_repetitive_families_have_surface_variation():
    factory = FactualRestraintArtifactFactory()

    ambiguous = [factory.build(1 + 8 * index).payload["question"] for index in range(92)]
    unannounced = [factory.build(3 + 8 * index).payload["question"] for index in range(92)]

    assert len({question.split(" ", 1)[0] for question in ambiguous}) > 1
    assert len({question.split(" ", 1)[0] for question in unannounced}) > 1
    assert not any("Systems's" in question for question in unannounced)


def test_general_vocabulary_context_uses_adjective_compatible_subjects_without_losing_variants():
    factory = EducationalQAMCQGeneralArtifactFactory()

    assert set(factory.VOCABULARY_SUBJECTS) == set(factory.ADJECTIVE_CONTEXT)
    assert all(len(subjects) == len(factory.OBJECTS) for subjects in factory.VOCABULARY_SUBJECTS.values())
    assert all(len(set(subjects)) == len(subjects) for subjects in factory.VOCABULARY_SUBJECTS.values())

    # Vocabulary is family index 2 in each eight-family cycle. Exercise every
    # adjective against every subject variant used in its deterministic cycle.
    rows = [factory.build(2 + 8 * index) for index in range(len(factory.ADJECTIVE_CONTEXT) * len(factory.OBJECTS))]
    assert {row.family for row in rows} == {"vocabulary"}

    for row in rows:
        question = row.payload["question"]
        evidence = row.payload["evidence"]
        adjective = question.split()[2]
        answer, clue = factory.ADJECTIVE_CONTEXT[adjective]
        subjects = factory.VOCABULARY_SUBJECTS[adjective]

        assert row.payload["choices"][row.payload["correct_index"]] == answer
        assert clue in evidence
        assert any(f"the {subject} as {adjective}" in evidence for subject in subjects)

    evidence = "\n".join(row.payload["evidence"] for row in rows)
    assert "spare key was narrow" not in evidence
    assert "spare key was careful" not in evidence
    assert "spare key was modest" not in evidence

    fingerprints = {artifact_fingerprint(row) for row in rows}
    assert len(fingerprints) == len(rows)


def test_all_grounded_generators_render_complete_batches():
    for signal in ("arithmetic", "task_code", "educational_qa_mcq_math", "educational_qa_mcq_general", "factual_restraint"):
        artifacts, records, telemetry = GroundedSignalGenerator(signal, GroundedMockLLM(), batch_size=32).generate_batch(0)
        assert len(artifacts) == len(records) == 32
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


def test_grounded_artifacts_have_no_placeholder_quality_failures():
    for factory in (EducationalQAMCQMathArtifactFactory, EducationalQAMCQGeneralArtifactFactory, FactualRestraintArtifactFactory):
        rows = [factory().build(index) for index in range(512)]
        assert all(not validate_artifact(row) for row in rows)
        assert len({artifact_fingerprint(row) for row in rows}) == len(rows)


def test_math_mcq_positive_quantity_families_have_nonnegative_plausible_choices():
    factory = EducationalQAMCQMathArtifactFactory()
    for index in range(500):
        artifact = factory.build(index)
        if artifact.family in {"missing_operand", "exact_division", "two_step_quantity"}:
            assert all(int(choice) >= 0 for choice in artifact.payload["choices"])


def test_batch_store_persists_telemetry(tmp_path):
    artifacts, records, _ = GroundedSignalGenerator("factual_restraint", GroundedMockLLM(), batch_size=32).generate_batch(0)
    store = GroundedBatchStore(tmp_path, "factual_restraint")
    store.write_completed(
        batch_id=0, artifacts=artifacts, records=records,
        telemetry={"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.01}, "elapsed_seconds": 1.5, "retry_count": 1},
    )
    assert store.telemetry_summary()["total_tokens"] == 30
    assert store.telemetry_summary()["cost"] == 0.01


def test_run_signal_supports_bounded_concurrent_grounded_batches(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 2},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 32, "samples": 64}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    generate.run_signal("factual_restraint", cfg, tmp_path)
    assert len((tmp_path / "raw" / "factual_restraint.jsonl").read_text().splitlines()) == 64
    assert len(list((tmp_path / "manifests" / "grounded" / "factual_restraint" / "batches").glob("batch_*.json"))) == 2


def test_run_signal_supports_batch_size_64_for_qualification(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 64, "parallel_requests": 8},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 64, "samples": 64}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    generate.run_signal("factual_restraint", cfg, tmp_path)
    assert len((tmp_path / "raw" / "factual_restraint.jsonl").read_text().splitlines()) == 64


def test_run_signal_rejects_batch_size_above_qualification_limit(tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 65, "parallel_requests": 8},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 65, "samples": 65}},
    }
    with pytest.raises(ValueError, match="batch_size between 1 and 64"):
        generate.run_signal("factual_restraint", cfg, tmp_path)


def test_run_signal_supports_concurrency_32_for_qualification(monkeypatch, tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 32},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 32, "samples": 64}},
    }
    monkeypatch.setattr(generate, "build_llm", lambda *args, **kwargs: GroundedMockLLM())
    generate.run_signal("factual_restraint", cfg, tmp_path)
    assert len((tmp_path / "raw" / "factual_restraint.jsonl").read_text().splitlines()) == 64


def test_run_signal_rejects_concurrency_above_qualification_limit(tmp_path):
    cfg = {
        "target_total_tokens": 5000,
        "backend": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"},
        "generation": {"batch_size": 32, "parallel_requests": 33},
        "mix": {"factual_restraint": {"architecture": "grounded", "batch_size": 32, "samples": 64}},
    }
    with pytest.raises(ValueError, match="parallel_requests between 1 and 32"):
        generate.run_signal("factual_restraint", cfg, tmp_path)
