import json
from typing import Any, Dict, List, Optional

# -------------------------------------------------------------------
# Batched wrapper templates
# -------------------------------------------------------------------
# Groq/OpenAI JSON object mode requires a top-level JSON object, not a
# bare JSON array. Both candidate and response passes ask for {"items": [...]}.

BATCHED_WRAPPER_TEMPLATE = """You are a synthetic data candidate generator.

Return ONLY valid JSON.

The top-level JSON value MUST be an object with this exact shape:

{{
  \"items\": [
    {{
      ... record fields ...
    }}
  ]
}}

Requirements:
- The \"items\" array MUST contain exactly {batch_size} objects.
- Each item MUST strictly follow this JSON Schema:

{schema}

Rules:
- Output ONLY the JSON object.
- No explanations outside JSON.
- No prose outside JSON.
- No markdown.
- No code fences.
- No comments.
- Every item MUST match the schema.
- Keep each field concise but not generic.
- Use plain JSON strings.
- Do not include raw unescaped newlines inside string values.
- Do not omit required fields.
- Do not repeat the same question, task, or wording within the batch.
- Do not copy examples from these instructions.
- Do not include the batch diversity id in any output field.

Candidate-generation task:
{task_instruction}

Diversity constraints for this batch:
{diversity_context}
"""

RESPONSE_WRAPPER_TEMPLATE = """You are the independent answering model for synthetic training data.

Return ONLY valid JSON.

The top-level JSON value MUST be an object with this exact shape:

{{
  \"items\": [
    {{
      ... response fields ...
    }}
  ]
}}

Requirements:
- The \"items\" array MUST contain exactly {batch_size} objects.
- Each response item MUST strictly follow this JSON Schema:

{schema}

Rules:
- Output ONLY the JSON object.
- No prose outside JSON.
- No markdown.
- No code fences.
- No comments.
- Return one response for every candidate_id exactly once.
- Answer each candidate independently from its supplied question or task.
- Do not trust an implied answer from a candidate; solve the task yourself.
- Keep fields concise and suitable for training data.

Response/completion task:
{task_instruction}

Candidates to answer:
{candidates}
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


def build_response_prompt(
    schema: Dict[str, Any],
    task_instruction: str,
    candidates: List[Dict[str, Any]],
) -> str:
    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
    candidate_json = json.dumps(candidates, indent=2, ensure_ascii=False)
    return RESPONSE_WRAPPER_TEMPLATE.format(
        batch_size=len(candidates),
        schema=schema_json,
        task_instruction=task_instruction.strip(),
        candidates=candidate_json,
    )
