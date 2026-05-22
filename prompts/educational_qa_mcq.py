EDU_QA_MCQ_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "educational_qa_mcq" },
    "question": { "type": "string" },
    "choices": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 4,
      "maxItems": 4
    },
    "correct_index": { "type": "integer", "minimum": 0, "maximum": 3 },
    "explanation": { "type": "string" },
    "verification_expression": { "type": "string" },
    "verification_answer": { "type": "string" }
  },
  "required": [
    "type",
    "question",
    "choices",
    "correct_index",
    "explanation",
    "verification_expression",
    "verification_answer"
  ]
}
"""

# Backward-compatible alias for older imports.
EDUCATIONAL_QA_MCQ_SCHEMA = EDU_QA_MCQ_SCHEMA

EDU_QA_MCQ_TASK = r"""Generate independent educational multiple-choice records with machine-verifiable integer answers.

Each raw generated item must have exactly these fields:
- "type": "educational_qa_mcq"
- "question": one clear numeric educational question
- "choices": exactly 4 distinct answer choices, each written as a plain integer string
- "correct_index": an integer from 0 to 3 pointing to the correct numeric choice
- "explanation": one short sentence showing the calculation and final answer
- "verification_expression": a plain arithmetic expression that computes the correct answer
- "verification_answer": the exact integer result of verification_expression, matching choices[correct_index]

The verification fields are temporary validation metadata and will not be published.

Allowed verified question families:
- integer addition, subtraction, multiplication, or exact integer division
- two-step integer arithmetic
- missing integer value where the answer is computable
- rectangle area with both length and width explicitly provided
- rectangle perimeter with both length and width explicitly provided
- exact fractional quantity problems whose answer is an integer
- exact percentage-of-quantity problems whose answer is an integer
- ratio-share questions whose answer is an integer
- totals, differences, or integer averages from a tiny described list or table

Verification-expression rules:
- Use only integer literals, spaces, parentheses, and the operators +, -, *, and /.
- Use / only when the final computed result is an exact integer.
- Do not use variables, units, percentages, fractions-as-text, exponentiation, functions, imports, or prose in verification_expression.
- verification_answer must be a plain integer string, with no units or explanation.
- choices must also be plain integer strings so the validator can compare them exactly.

Correctness rules:
- Compute the answer before selecting correct_index.
- The indexed choice must equal verification_answer.
- The explanation must state the same final numeric answer and briefly show why it is correct.
- All distractor choices must be distinct and numerically incorrect.
- Include every quantity needed to solve the question.
- For rectangle questions, always give both length and width.
- For fraction and percent questions, choose values that yield an exact integer answer.
- For average questions, choose values whose average is an integer.

Disallowed question families:
- open-ended "what should happen next" questions
- "best explanation" or opinion-based questions
- arbitrary spending or decision-choice scenarios
- questions asking how to increase or decrease an area or perimeter
- geometry questions missing required dimensions
- conceptual definitions, trivia, history, geography, science facts, health advice, or current facts
- Python or programming questions
- questions whose correct answer cannot be verified by the numeric expression
- numeric situations that result in fractional people, objects, shots, books, pages, or other countable items

Output quality rules:
- Keep each question self-contained.
- Use concise wording.
- Use the per-batch verified profile as a hard constraint.
- Do not copy concrete values or phrasing from these instructions.
- Do not repeat question wording, operand combinations, answers, or distractor sets within the batch.
- Vary correct_index across records.
- Do not include any extra top-level keys.
"""

# Backward-compatible alias for older imports. This is the task text, not schema.
EDUCATIONAL_QA_MCQ_TASK = EDU_QA_MCQ_TASK


def build_educational_qa_mcq_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE

    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=EDU_QA_MCQ_SCHEMA,
        task_instruction=EDU_QA_MCQ_TASK,
        diversity_context=(
            "Generate one deterministic numeric MCQ with a safe arithmetic "
            "verification_expression and an exact integer verification_answer."
        ),
    )
