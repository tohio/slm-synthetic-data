from slm_synth.record_quality import validate_record


def test_math_mcq_derives_answer_index_and_strips_verification_metadata():
    raw = {
        "type": "educational_qa_mcq_math",
        "question": "What is 25% of 120?",
        "choices": ["15", "20", "30", "25"],
        "correct_index": 3,
        "explanation": "25% of 120 is 30 because (120 * 25) / 100 = 30.",
        "verification_expression": "(120 * 25) / 100",
        "verification_answer": "30",
    }
    result = validate_record("educational_qa_mcq_math", raw, require_mcq_verification=True)
    assert result.ok
    assert result.record is not None
    assert result.record["correct_index"] == 2
    assert "verification_expression" not in result.record
    assert "verification_answer" not in result.record


def test_math_mcq_rejects_wrong_verified_answer():
    raw = {
        "type": "educational_qa_mcq_math",
        "question": "What is 8 times 5?",
        "choices": ["30", "35", "40", "45"],
        "correct_index": 2,
        "explanation": "Eight times five is 40.",
        "verification_expression": "8 * 5",
        "verification_answer": "45",
    }
    result = validate_record("educational_qa_mcq_math", raw, require_mcq_verification=True)
    assert not result.ok
    assert "mcq_verification_answer_mismatch" in result.issues


def test_general_mcq_uses_regular_mcq_schema_without_verification():
    row = {
        "type": "educational_qa_mcq_general",
        "question": "A rule says never share a password. Which action follows the rule?",
        "choices": ["Email it", "Post it", "Keep it private", "Text it"],
        "correct_index": 2,
        "explanation": "Keeping it private follows the stated rule.",
    }
    result = validate_record("educational_qa_mcq_general", row)
    assert result.ok


def test_math_mcq_rejects_meta_commentary_in_explanation():
    raw = {
        "type": "educational_qa_mcq_math",
        "question": "A shelf has 24 books packed into boxes of 4. How many boxes are needed?",
        "choices": ["4", "6", "8", "12"],
        "correct_index": 1,
        "explanation": "24 / 4 = 6, but the provided choices suggest a different interpretation and an error in the question generation.",
        "verification_expression": "24 / 4",
        "verification_answer": "6",
    }
    result = validate_record("educational_qa_mcq_math", raw, require_mcq_verification=True)
    assert not result.ok
    assert "mcq_math_meta_commentary" in result.issues


def test_arithmetic_verifies_answer_and_strips_temporary_metadata():
    raw = {
        "type": "arithmetic",
        "question": "What is 19 multiplied by 654?",
        "steps": ["19 * 654 = 12426"],
        "answer": "12426",
        "verification_expression": "19 * 654",
        "verification_answer": "12426",
    }
    result = validate_record("arithmetic", raw, require_arithmetic_verification=True)
    assert result.ok
    assert result.record is not None
    assert result.record["answer"] == "12426"
    assert "verification_expression" not in result.record
    assert "verification_answer" not in result.record


def test_arithmetic_rejects_wrong_verified_answer():
    raw = {
        "type": "arithmetic",
        "question": "What is 19 multiplied by 654?",
        "steps": ["19 * 654 = 12431"],
        "answer": "12431",
        "verification_expression": "19 * 654",
        "verification_answer": "12431",
    }
    result = validate_record("arithmetic", raw, require_arithmetic_verification=True)
    assert not result.ok
    assert "arithmetic_verification_answer_mismatch" in result.issues


def test_arithmetic_rejects_response_meta_commentary():
    raw = {
        "type": "arithmetic",
        "question": "A garage has 70 occupied spaces and 14 cars leave. How many spaces remain occupied?",
        "steps": ["Without knowing initial occupancy, assuming all spaces were occupied, 70 - 14 = 56."],
        "answer": "56",
        "verification_expression": "70 - 14",
        "verification_answer": "56",
    }
    result = validate_record("arithmetic", raw, require_arithmetic_verification=True)
    assert not result.ok
    assert "arithmetic_meta_commentary" in result.issues
