import json

# -------------------------------------------------------------------
# Unified wrapper template (no YAML parsing, no external loading)
# -------------------------------------------------------------------

WRAPPER_TEMPLATE = """You are a JSON-only generator.

You MUST output a single JSON object that strictly follows this JSON Schema:

{schema}

Rules:
- Output ONLY valid JSON.
- No explanations.
- No prose.
- No markdown.
- No code fences.
- No comments.
- No natural language outside JSON.
- Arrays MUST be arrays, never strings.
- All required fields MUST be present.
- All fields MUST match the schema types.
- If unsure, output an empty array or empty string, but NEVER omit fields.

Now follow this task description and produce ONE JSON object:

{task_instruction}
"""

# -------------------------------------------------------------------
# Unified prompt builder (schema + task text only)
# -------------------------------------------------------------------

def build_prompt(schema: dict, task_instruction: str, prompt_name: str | None = None) -> str:
    """
    Build the final LLM prompt.

    - schema: Python dict (will be JSON-encoded)
    - task_instruction: plain text describing what to generate
    - prompt_name: ignored (kept for compatibility with generators)
    """
    schema_json = json.dumps(schema, indent=2)

    return WRAPPER_TEMPLATE.format(
        schema=schema_json,
        task_instruction=task_instruction.strip(),
    )
