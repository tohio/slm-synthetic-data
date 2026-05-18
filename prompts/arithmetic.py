from .wrapper import WRAPPER_TEMPLATE

ARITHMETIC_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "arithmetic" },
    "question": { "type": "string" },
    "answer": { "type": "number" },
    "steps": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["type", "question", "answer", "steps"]
}
"""

ARITHMETIC_TASK = """Generate a single arithmetic reasoning problem.

The JSON object MUST have:
- "type": "arithmetic"
- "question": a short math question as a string
- "answer": the numeric answer
- "steps": an array of short strings, each describing one reasoning step
"""

def build_arithmetic_prompt() -> str:
    return WRAPPER_TEMPLATE.format(
        schema=ARITHMETIC_SCHEMA,
        task_instruction=ARITHMETIC_TASK,
    )
