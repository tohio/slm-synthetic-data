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

TASK_CODE_TASK = """Generate independent beginner programming task records.

Each item must have:
- "type": "task_code"
- "task": a short programming task
- "plan": 2 to 4 short solution steps
- "code": a short Python code snippet solving the task

Rules for "code":
- Use Python only.
- Keep code under 20 lines.
- Do not use markdown fences.
- Do not include triple backticks.
- Avoid very long strings.
- JSON must escape newline characters correctly inside the code string.
"""


def build_task_code_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=TASK_CODE_SCHEMA,
        task_instruction=TASK_CODE_TASK,
    )
