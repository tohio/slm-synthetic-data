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
- "verification_answer": the exact final integer answer requested by the question, as a string

Final-answer verification rules:
- Solve the complete question privately before writing any output fields.
- verification_expression MUST compute the final answer requested by the question in one complete expression, not an intermediate subtotal, sum, product, numerator, or denominator.
- verification_answer MUST equal the final result of verification_expression.
- The final numeric result stated in explanation MUST exactly equal verification_answer.
- Before returning each item, compare the final result in explanation to verification_answer. If they differ, correct verification_expression and verification_answer before returning.
- For mean or average questions, verification_expression MUST include division by the number of values. Example form: (48 + 60 + 72) / 3, never only 48 + 60 + 72.
- For percentage, fraction, or ratio questions, verification_expression MUST calculate the requested final share, remainder, or total, not an intermediate conversion.
- If the supplied question does not have one exact integer final answer, return empty strings for explanation, verification_expression, and verification_answer so the item is rejected locally. Do not round or invent an answer.

Rules:
- Solve from the supplied question, not from its candidate choices.
- Do not return question, choices, or correct_index. Python assembles those fields from your solved final answer.
- Do not change operands, quantities, units, or wording from the supplied question.
- Use only integer literals, spaces, parentheses, and +, -, *, / in verification_expression.
- Use division only when the final answer is an exact integer.
- Do not mention corrections, errors, options, answer keys, or the generation process.
"""
