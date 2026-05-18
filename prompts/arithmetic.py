
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
- "answer": the final answer as a string
- "steps": 1 to 3 short strings explaining the calculation

Allowed task forms:
- direct equation
- word problem
- missing-value equation
- compare two computed values
- two-step integer arithmetic
- reverse-check problem

Rules:
- Use only addition, subtraction, multiplication, or exact integer division.
- Use varied numbers, varied wording, and varied answer values.
- Avoid trivial repeated examples such as "2 + 2".
- For exact division, choose numbers that divide evenly.
- Keep questions short and unambiguous.
"""


def build_arithmetic_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=ARITHMETIC_SCHEMA,
        task_instruction=ARITHMETIC_TASK,
        diversity_context="Use a varied arithmetic format, operation, and number range.",
    )
