from slm_synth.schemas import (
    validate_arithmetic,
    validate_educational_qa_mcq_general,
    validate_educational_qa_mcq_math,
    validate_factual_restraint,
    validate_task_code,
)


def test_arithmetic_schema_valid():
    validate_arithmetic(
        {
            "type": "arithmetic",
            "question": "What is 2 + 2?",
            "steps": ["2 + 2 = 4"],
            "answer": "4",
        }
    )


def test_task_code_schema_valid():
    validate_task_code(
        {
            "type": "task_code",
            "task": "Sort a list",
            "plan": ["Read list", "Sort list", "Return result"],
            "code": "def sort_values(values):\n    return sorted(values)",
        }
    )


def test_educational_qa_mcq_math_raw_schema_valid():
    validate_educational_qa_mcq_math(
        {
            "type": "educational_qa_mcq_math",
            "question": "A rectangle is 8 cm long and 5 cm wide. What is its area?",
            "choices": ["30", "35", "40", "45"],
            "correct_index": 2,
            "explanation": "The area is 8 * 5 = 40 square centimeters.",
            "verification_expression": "8 * 5",
            "verification_answer": "40",
        }
    )


def test_educational_qa_mcq_general_schema_valid():
    validate_educational_qa_mcq_general(
        {
            "type": "educational_qa_mcq_general",
            "evidence": "Sentence: Mina quickly packed the box.",
            "question": "In the sentence 'Mina quickly packed the box,' which word is an adverb?",
            "choices": ["Mina", "quickly", "packed", "box"],
            "correct_index": 1,
            "explanation": "'Quickly' describes how Mina packed the box.",
        }
    )


def test_factual_restraint_schema_valid():
    validate_factual_restraint(
        {
            "type": "factual_restraint",
            "question": "Who will win the next World Cup?",
            "safe_answer": "I cannot predict future events.",
        }
    )
