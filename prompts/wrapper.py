# -------------------------------------------------------------------
# Batched wrapper template (JSON array of N objects)
# -------------------------------------------------------------------

BATCHED_WRAPPER_TEMPLATE = """You are a JSON-only generator.

Produce EXACTLY {batch_size} JSON objects in a SINGLE JSON array.
Each element MUST strictly follow this JSON Schema:

{schema}

Rules:
- Output ONLY a JSON array.
- No explanations.
- No prose.
- No markdown.
- No code fences.
- No comments.
- No natural language outside JSON.
- Every element MUST match the schema.
- If unsure, output an empty string or empty array, but NEVER omit fields.

Task:
{task_instruction}
"""

def build_batched_prompt(schema: dict, task_instruction: str, batch_size: int) -> str:
    schema_json = json.dumps(schema, indent=2)
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=batch_size,
        schema=schema_json,
        task_instruction=task_instruction.strip(),
    )
