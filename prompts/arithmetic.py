
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
- Use concise natural language."""


def build_arithmetic_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=ARITHMETIC_SCHEMA,
        task_instruction=ARITHMETIC_TASK,
        diversity_context="Use a varied arithmetic format, operation, and number range.",
    )
