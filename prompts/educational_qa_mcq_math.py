EDUCATIONAL_QA_MCQ_MATH_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "educational_qa_mcq_math" },
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

EDUCATIONAL_QA_MCQ_MATH_TASK = r"""Generate independent mathematical multiple-choice records with machine-verifiable integer answers.

Each raw generated item must have exactly these fields:
- "type": "educational_qa_mcq_math"
- "question": one clear numeric educational question
- "choices": exactly 4 distinct answer choices, each written as a plain integer string
- "correct_index": the actual position of the correct numeric choice after the choices are written
- "explanation": one short sentence showing the calculation and final answer
- "verification_expression": a plain arithmetic expression that computes the correct answer
- "verification_answer": the exact integer result of verification_expression, matching one choice

The verification fields are temporary validation metadata and will not be published. Validation independently derives the final correct_index from the unique verified answer choice.

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
- Do not use variables, units, percentages, decimals, exponentiation, functions, imports, or prose in verification_expression.
- verification_answer must be a plain integer string, with no units or explanation.
- choices must also be plain integer strings so the validator can compare them exactly.

Correctness rules:
- Compute verification_expression and verification_answer before building choices.
- Include verification_answer exactly once in choices.
- Create three distinct numerically incorrect distractors.
- Set correct_index to the actual position containing verification_answer; do not target a predetermined position.
- Include every quantity used in verification_expression in the question.
- Ensure the nouns in the question match the quantity being computed: do not ask for slices when the supplied total is people, or boxes when the supplied total is books.
- The explanation must state the same final numeric answer and briefly show the same computation.

Explanation rules:
- Explain only the calculation and result.
- Do not mention the options, provided choices, answer-key selection, question generation, an error, a correction, or a different interpretation.
- Do not include self-critique, apologies, or commentary about whether the question is valid.

Disallowed question families:
- open-ended "what should happen next" questions
- "best explanation" or opinion-based questions
- arbitrary spending or decision-choice scenarios
- questions asking how to increase or decrease an area or perimeter
- geometry questions missing required dimensions
- conceptual definitions, trivia, history, geography, science facts, health advice, or current facts
- Python or programming questions
- numeric situations that result in fractional countable items

Output quality rules:
- Keep each question self-contained and concise.
- Use the per-batch verified profile as a hard constraint.
- Do not copy concrete values or phrasing from these instructions.
- Do not repeat question wording, operand combinations, answers, or distractor sets within the batch.
- Do not include any extra top-level keys.
"""


def build_educational_qa_mcq_math_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE

    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=EDUCATIONAL_QA_MCQ_MATH_SCHEMA,
        task_instruction=EDUCATIONAL_QA_MCQ_MATH_TASK,
        diversity_context=(
            "Generate one deterministic math MCQ with a safe arithmetic "
            "verification_expression and an exact integer verification_answer. "
            "Do not target a predetermined answer position and do not write "
            "meta-commentary in the explanation."
        ),
    )
