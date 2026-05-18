import yaml
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PROMPT_DIR = ROOT / "prompts"

WRAPPER_TEMPLATE = """You are a JSON-only generator.

You MUST output a single JSON object that strictly follows this JSON Schema:

{schema_json}

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

def load_prompt_yaml(name: str) -> dict:
    path = PROMPT_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return yaml.safe_load(path.read_text())

def build_prompt(schema: dict, task_instruction: str, prompt_name: str) -> str:
    prompt_yaml = load_prompt_yaml(prompt_name)
    task_text = prompt_yaml.get("task_instruction", "") + "\n" + task_instruction

    # FIX: Convert schema dict → pretty JSON string
    schema_json = json.dumps(schema, indent=2)

    return WRAPPER_TEMPLATE.format(
        schema_json=schema_json,
        task_instruction=task_text.strip()
    )
