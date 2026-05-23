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
        "explanation": {"type": "string"},
        "verification_expression": {"type": "string"},
        "verification_answer": {"type": "string"},
    },
    "required": [
        "candidate_id", "explanation", "verification_expression", "verification_answer"
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

EDUCATIONAL_QA_MCQ_MATH_RESPONSE_TASK = r"""Independently solve each mathematical MCQ candidate.

For each candidate_id, return only the answer-side fields:
- "candidate_id": the supplied id
- "explanation": one concise calculation statement with the final numeric answer
- "verification_expression": a plain integer arithmetic expression matching the supplied question
- "verification_answer": the exact integer result as a string

Rules:
- Solve from the supplied question, not from its candidate choices.
- Do not return question, choices, or correct_index. Python assembles those fields from your solved answer.
- Do not change operands, quantities, units, or wording from the supplied question.
- Use only integer literals, spaces, parentheses, and +, -, *, / in verification_expression.
- Use division only when the final answer is an exact integer.
- Do not mention corrections, errors, options, answer keys, or the generation process.
"""
