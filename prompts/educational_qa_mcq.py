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
- "correct_index": an integer from 0 to 3 pointing to the one correct choice
- "explanation": one short sentence showing the calculation and final integer answer
- "verification_expression": a plain integer arithmetic expression that computes the correct answer
- "verification_answer": the exact integer result of verification_expression

The verification fields are temporary validation metadata and will not be published.

Use only these verified question families:
- integer addition or subtraction
- integer multiplication
- exact integer division with no remainder
- two-step integer arithmetic using addition and/or subtraction
- rectangle area with both integer length and integer width explicitly stated
- rectangle perimeter with both integer length and integer width explicitly stated
- one-half of an even integer total
- one-quarter or three-quarters of a total divisible by 4
- 10%, 20%, 25%, 50%, or 75% of a total chosen so the result is an integer
- total or difference from three explicitly stated integer values
- average of three explicitly stated integer values whose sum is divisible by 3

Do NOT generate:
- ratio-share questions
- general fraction questions outside 1/2, 1/4, or 3/4
- percentages other than 10%, 20%, 25%, 50%, or 75%
- decimals, money amounts involving cents, rounding, approximations, or remainders
- open-ended "what should happen next" questions
- "best explanation", opinion, decision-choice, or conceptual-definition questions
- questions about changing an area or perimeter
- geometry questions missing required dimensions
- trivia, science facts, history, geography, reading, health, or current facts
- Python or programming questions
- questions whose answer cannot be verified from a plain integer arithmetic expression
- countable-item situations that would produce a fractional answer

Verification-expression rules:
- Use only integer literals, spaces, parentheses, and the operators +, -, *, and /.
- Never use decimal literals.
- Use / only when the computed result is an exact integer.
- For percentages, encode the calculation using integers, for example `(total * percent) / 100`; do not use decimals.
- For fractions, encode the calculation using integers, for example `(total * numerator) / denominator`.
- Do not use variables, units, percentages, fraction notation outside the expression arithmetic, exponentiation, functions, imports, or prose in verification_expression.
- verification_answer must be a plain integer string with no units or explanation.

Mandatory answer-construction procedure:
1. Write the self-contained question using the assigned verified family.
2. Build verification_expression from the exact quantities stated in the question.
3. Compute the exact integer result and write it as verification_answer.
4. Build choices only after computing verification_answer.
5. Put verification_answer in choices exactly once.
6. Create three distinct nearby incorrect integer choices; none may equal verification_answer and no two choices may be equal.
7. Set correct_index to the position containing verification_answer.
8. Write an explanation that includes the same final integer answer and the same calculation.

Choice rules:
- All four choices must be distinct plain integer strings.
- Exactly one choice must equal verification_answer.
- Do not omit the verified answer from choices.
- Do not place the verified answer in choices more than once.
- Do not use units, decimals, fractions, percent signs, words, or punctuation in choices.
- Before returning the JSON object, re-check that `choices[correct_index] == verification_answer`.

Quality rules:
- Keep each question self-contained and concise.
- Use the per-batch verified profile as a hard constraint.
- Include every number needed to solve the question.
- Do not copy concrete values or wording from these instructions.
- Use varied operands, contexts, distractors, and answer positions across records.
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
            "Generate one restricted verified numeric MCQ. Compute the exact integer "
            "answer first, include it exactly once in four distinct integer choices, "
            "and do not generate ratios, arbitrary fractions, decimals, or rounding."
        ),
    )
