import json

from slm_synth.dedup import dedup_signal


def test_exact_dedup_general_mcq(tmp_path):
    validated = tmp_path / "validated"
    deduped = tmp_path / "deduped"
    validated.mkdir()
    record = {
        "type": "educational_qa_mcq_general",
        "question": "Which word is an adverb in 'Mina quickly packed the box'?",
        "choices": ["Mina", "quickly", "packed", "box"],
        "correct_index": 1,
        "explanation": "Quickly describes the action.",
    }
    path = validated / "educational_qa_mcq_general.jsonl"
    with path.open("w") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.write(json.dumps(record) + "\n")

    kept, dropped = dedup_signal(validated, deduped, "educational_qa_mcq_general")
    assert kept == 1
    assert dropped == 1


def test_math_mcq_dedup_ignores_choice_permutations_for_same_question_and_answer(tmp_path):
    validated = tmp_path / "validated"
    deduped = tmp_path / "deduped"
    validated.mkdir()
    rows = [
        {
            "type": "educational_qa_mcq_math",
            "question": "Start with 72, remove 42, then remove 10. How many remain?",
            "choices": ["20", "18", "16", "22"],
            "correct_index": 0,
            "explanation": "72 - 42 = 30, then 30 - 10 = 20.",
        },
        {
            "type": "educational_qa_mcq_math",
            "question": "Start with 72, remove 42, then remove 10. How many remain?",
            "choices": ["18", "20", "16", "22"],
            "correct_index": 1,
            "explanation": "72 - 42 = 30, then 30 - 10 = 20.",
        },
    ]
    path = validated / "educational_qa_mcq_math.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    kept, dropped = dedup_signal(validated, deduped, "educational_qa_mcq_math")
    assert kept == 1
    assert dropped == 1


def test_two_step_math_artifacts_carry_required_item_terms_and_prompt_requires_them():
    from slm_synth.artifacts import EducationalQAMCQMathArtifactFactory
    from slm_synth.grounded import GroundedSignalGenerator

    class NoopLLM:
        pass

    factory = EducationalQAMCQMathArtifactFactory()
    artifacts = [factory.build(3 + 5 * index) for index in range(len(factory.SETTINGS))]
    assert all(artifact.family == "two_step_quantity" for artifact in artifacts)
    assert {artifact.payload["required_text_literals"][0] for artifact in artifacts} == set(factory.SETTINGS)

    prompt = GroundedSignalGenerator(
        "educational_qa_mcq_math", NoopLLM(), batch_size=len(artifacts), factory=factory
    ).build_prompt(artifacts)
    assert "preserve each supplied term in the question exactly" in prompt
    assert "required_text_literals" in prompt
