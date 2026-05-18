
import json
from typing import Any, Dict, Optional

# -------------------------------------------------------------------
# Batched wrapper template
# -------------------------------------------------------------------
# Groq/OpenAI JSON object mode requires a top-level JSON object, not a
# bare JSON array. The generator therefore asks for {"items": [...]}.

BATCHED_WRAPPER_TEMPLATE = """You are a synthetic data generator.

Return ONLY valid JSON.

The top-level JSON value MUST be an object with this exact shape:

{{
  "items": [
    {{
      ... record fields ...
    }}
  ]
}}

Requirements:
- The "items" array MUST contain exactly {batch_size} objects.
- Each item MUST strictly follow this JSON Schema:

{schema}

Rules:
- Output ONLY the JSON object.
- No explanations.
- No prose.
- No markdown.
- No code fences.
- No comments.
- No natural language outside JSON.
- Every item MUST match the schema.
- Keep each field concise but not generic.
- Use plain JSON strings.
- Do not include raw unescaped newlines inside string values.
- If unsure, use an empty string or empty array, but NEVER omit required fields.
- Do not repeat the same question, task, answer, function name, or wording within the batch.
- Do not copy examples from these instructions.
- Do not include the batch diversity id in any output field.

Generation task:
{task_instruction}

Diversity constraints for this batch:
{diversity_context}
"""


def build_batched_prompt(
    schema: Dict[str, Any],
    task_instruction: str,
    batch_size: int,
    diversity_context: Optional[str] = None,
) -> str:
    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=batch_size,
        schema=schema_json,
        task_instruction=task_instruction.strip(),
        diversity_context=(diversity_context or "Use broad variety across records.").strip(),
    )
