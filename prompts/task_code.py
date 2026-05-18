TASK_CODE_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "task_code" },
    "task": { "type": "string" },
    "plan": {
      "type": "array",
      "items": { "type": "string" }
    },
    "code": { "type": "string" }
  },
  "required": ["type", "task", "plan", "code"]
}
"""

TASK_CODE_TASK = """Generate a single programming task with a solution.

The JSON object MUST have:
- "type": "task_code"
- "task": a natural language description of the coding task
- "plan": an array of short strings, each describing one step in the solution plan
- "code": a code snippet solving the task
"""

def build_task_code_prompt() -> str:
    return WRAPPER_TEMPLATE.format(
        schema=TASK_CODE_SCHEMA,
        task_instruction=TASK_CODE_TASK,
    )
