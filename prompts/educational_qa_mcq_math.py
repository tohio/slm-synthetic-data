EDUCATIONAL_QA_MCQ_MATH_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "educational_qa_mcq_math_candidate"},
        "question": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": ["type", "question", "choices"],
}

EDUCATIONAL_QA_MCQ_MATH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "integer"},
        "question": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
        "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
        "explanation": {"type": "string"},
        "verification_expression": {"type": "string"},
        "verification_answer": {"type": "string"},
    },
    "required": [
        "candidate_id", "question", "choices", "correct_index", "explanation",
        "verification_expression", "verification_answer"
    ],
}

EDUCATIONAL_QA_MCQ_MATH_CANDIDATE_TASK = r"""Generate mathematical multiple-choice candidates without an answer key.

Each item must contain only:
- "type": "educational_qa_mcq_math_candidate"
- "question": one clear, self-contained numeric question
- "choices": four distinct plain integer-string candidate choices

Allowed families:
- integer addition, subtraction, multiplication, or exact integer division
- two-step integer arithmetic
- rectangle area or perimeter with dimensions supplied
- exact fractional or percentage quantity with an integer result
- exact ratio share with an integer result
- total, difference, or integer mean from supplied integers

Rules:
- Do not include correct_index, explanation, solution steps, verification_expression, or verification_answer.
- The question must state every quantity needed to solve it.
- Use integer-answer questions only; no rounding or ambiguous units.
- Do not signal which choice is correct.
"""

EDUCATIONAL_QA_MCQ_MATH_RESPONSE_TASK = r"""Independently solve and finalize each mathematical MCQ candidate.

For each candidate_id, return a complete corrected answer record containing:
- "candidate_id": the supplied id
- "question": preserve the supplied question unless a minimal wording repair is required for clarity
- "choices": preserve usable choices, but replace a choice when needed so the correct answer appears exactly once among four distinct plain integer strings
- "correct_index": the actual position of the correct answer in the returned choices
- "explanation": one concise calculation statement with the final numeric answer
- "verification_expression": a plain integer arithmetic expression matching the returned question
- "verification_answer": the exact integer result as a string

Rules:
- Solve from the question, not from the supplied choices.
- Ensure returned question, choices, index, expression, answer, and explanation all agree.
- Do not change operands to make a distractor correct.
- Do not mention corrections, errors, options, answer keys, or the generation process.
"""
