
ARITHMETIC_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "arithmetic" },
    "question": { "type": "string" },
    "answer": { "type": "string" },
    "steps": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["type", "question", "answer", "steps"]
}
"""

ARITHMETIC_TASK = """Generate independent arithmetic reasoning records.

Each item must have:
- "type": "arithmetic"
- "question": a short integer arithmetic question
- "steps": 2 to 4 short strings explaining the calculation
- "answer": the final answer as a string

Diversity requirements:
- Use the supplied diversity context, batch nonce, topic, difficulty, and format hints.
- Do not generate generic toy examples such as "2 + 2 = ?", "2 + 5 = ?", "What is 2 + 2?", or repeated "Add X and Y" prompts.
- Do not repeat the same question wording within a batch.
- Avoid repeating the same number pairs, same operation sequence, or same answer across items in the same batch.
- Use varied integer ranges:
  - small: 1 to 99
  - medium: 100 to 9,999
  - large: 10,000 to 999,999
  - include negative integers in some subtraction/comparison items
- Use exact integer division only; choose numbers so the answer has no remainder.
- Mix formats across the batch:
  - direct equation
  - short word problem
  - missing operand
  - compare two expressions
  - order three computed values
  - two-step arithmetic
  - rate/quantity word problem
  - inventory or accounting-style word problem
  - distance/time or unit-count word problem
  - grouped multiplication/addition problem
- For two-step items, keep the arithmetic simple enough to verify but make the wording distinct.
- Keep all questions self-contained.
- Do not use decimals, fractions, percentages, or algebra beyond a single missing integer.
- The answer must be only the final numeric result as a string.
- The steps should explain the calculation, not just restate the question.

Quality requirements:
- The calculation must be correct.
- The answer must match the final step.
- The problem should be clear without external context.
- Use concise natural language.

Additional generation requirements for diversity and scale:
- Keep arithmetic LLM-generated, but avoid template collapse.
- Vary problem type across the batch: one-step arithmetic, two-step arithmetic, missing operand, compare two expressions, order values, word problem, unit/rate/money/inventory scenario.
- Vary operation families across the batch: addition, subtraction, multiplication, integer division, and mixed operations.
- Vary number ranges across the batch: small integers, two-digit integers, three-digit integers, and occasional four-digit totals when appropriate.
- Do not repeat the same operands, same question stem, same operation pattern, or same answer within a batch.
- Prefer concrete contexts when using word problems: inventory, distance, rate, schedules, money, classroom counts, packages, tickets, measurements, or simple resource planning.
- For missing-operand examples, make the missing value clear and ensure the answer is the missing operand.
- For comparison/order examples, the answer should be the selected value/expression, not a long explanation.
- Steps must be compact and faithful to the calculation.
- Answers must be exact integer strings.
- Do not include fractions, decimals, remainders, algebra variables beyond a single missing value, or ambiguous wording.
"""


def build_arithmetic_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=ARITHMETIC_SCHEMA,
        task_instruction=ARITHMETIC_TASK,
        diversity_context="Use a varied arithmetic format, operation, and number range.",
    )
