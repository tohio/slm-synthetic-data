ARITHMETIC_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "arithmetic_candidate"},
        "question": {"type": "string"},
    },
    "required": ["type", "question"],
}

ARITHMETIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "integer"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"},
    },
    "required": ["candidate_id", "steps", "answer"],
}

ARITHMETIC_CANDIDATE_TASK = r"""Generate independent unsolved integer-arithmetic questions.

Each item must contain only:
- "type": "arithmetic_candidate"
- "question": a self-contained arithmetic question with no answer or solution steps

Allowed forms:
- direct equation with uncommon operands
- short word problem
- missing integer operand
- compare or order computed values
- two-step quantity problem
- exact integer division or allocation problem

Rules:
- Use only integer arithmetic: addition, subtraction, multiplication, and exact division.
- Do not include decimals, fractions, percentages, remainders, or ambiguous wording.
- Do not state, hint at, or embed the answer in the question.
- Avoid repeated toy templates and repeated operands.
"""

ARITHMETIC_RESPONSE_TASK = r"""Solve each fixed arithmetic candidate independently.

For each candidate_id, return only:
- "candidate_id": the supplied id
- "steps": 2 to 4 compact calculation-focused strings
- "answer": the exact final integer as a string

Rules:
- Answer the supplied question exactly as written.
- Do not alter numbers or silently repair the question.
- If the item asks for a missing value, answer with that missing integer.
- Keep steps concise; do not include meta-commentary or discuss the answer choices.
"""
