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
        "verification_expression": {"type": "string"},
        "verification_answer": {"type": "string"},
    },
    "required": [
        "candidate_id",
        "steps",
        "answer",
        "verification_expression",
        "verification_answer",
    ],
}

ARITHMETIC_CANDIDATE_TASK = r"""Generate independent unsolved integer-arithmetic questions.

Each item must contain only:
- "type": "arithmetic_candidate"
- "question": a self-contained arithmetic question with no answer or solution steps

Allowed forms:
- direct equation with uncommon operands
- short word problem with one requested integer result
- missing integer operand
- comparison of computed values only when asking for the unique largest or smallest numeric value
- two-step quantity problem
- exact integer division or allocation problem

Coherence requirements:
- Generate only one coherent question with one uniquely determined integer answer.
- Before returning a candidate, privately solve it and confirm that every stated quantity is required by the calculation.
- Do not include irrelevant quantities, conflicting facts, or a story detail that is unused in the answer.
- Do not require an unstated assumption, such as initial occupancy, an omitted price, or an omitted group count.
- If comparing computed values, ask only for the unique largest or smallest numeric value, ensure there is no tie, and never ask which person, route, item, or category produced it.
- For occupancy or remaining-capacity questions, explicitly state the starting occupied/available quantity; never assume a lot, garage, tank, or container begins full or empty.
- For capacity questions, never add contents beyond stated capacity unless the question explicitly asks about overflow rather than contents.
- For totals after packing, shelving, or allocating items, omit the intermediate packing/allocation detail unless it changes the requested final quantity.
- Ask for a numeric answer only. Do not ask "who", "which item", "which route", or "which group" when the output is expected to be an integer.
- Do not create budget or cost problems unless every required price, budget, and prior expenditure is explicitly stated and necessary to calculate the requested remaining amount.
- If any rule fails, replace the candidate before returning the batch.

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
- "verification_expression": one complete integer arithmetic expression that computes the requested final answer
- "verification_answer": the exact final integer result of verification_expression as a string

Answer-verification requirements:
- Solve exactly the question stated by the candidate and verify the arithmetic before returning.
- verification_expression MUST compute the final requested answer, not an intermediate subtotal.
- answer MUST exactly equal verification_answer, and the final result stated in steps MUST equal both fields.
- Use only integer literals, spaces, parentheses, and +, -, *, and / in verification_expression.
- Use division only when the final result is an exact integer.
- For a missing-value problem, verification_expression must compute the missing integer.
- For a comparison problem, answer only if the question explicitly asks for one unique numeric largest or smallest value; reject any comparison asking which named person, route, category, item, or group wins.
- Before solving a word problem, confirm that every stated quantity is necessary and that no initial quantity, price, occupancy, capacity rule, or group count is missing.
- Reject any occupancy question that does not state the initial occupied or available amount, any capacity question whose requested contents exceed stated capacity, and any scenario containing irrelevant operations that do not affect the requested answer.
- If the candidate is ambiguous, contains unused or conflicting quantities, requires an unstated assumption, asks for an entity instead of a number, or has multiple valid answers, return empty strings for answer, verification_expression, and verification_answer and an empty steps array so it is rejected locally.

Rules:
- Answer the supplied question exactly as written.
- Do not alter numbers or silently repair the question.
- Keep steps concise and calculation-focused; do not include assumptions, self-corrections, or meta-commentary.
"""
