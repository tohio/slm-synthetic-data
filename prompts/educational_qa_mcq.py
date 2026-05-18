
EDU_QA_MCQ_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "educational_qa_mcq" },
    "question": { "type": "string" },
    "choices": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 4,
      "maxItems": 4
    },
    "correct_index": { "type": "integer" },
    "explanation": { "type": "string" }
  },
  "required": ["type", "question", "choices", "correct_index", "explanation"]
}
"""

# Backward-compatible alias for older imports.
EDUCATIONAL_QA_MCQ_SCHEMA = EDU_QA_MCQ_SCHEMA

EDU_QA_MCQ_TASK = """Generate independent educational multiple-choice records.

Each item must have:
- "type": "educational_qa_mcq"
- "question": one clear educational question
- "choices": exactly 4 short answer choices
- "correct_index": an integer from 0 to 3
- "explanation": one short sentence explaining the correct answer

Allowed subjects include:
- arithmetic
- basic algebra
- geometry
- physical science
- earth science
- biology basics
- computer science basics
- Python concepts
- history
- geography
- grammar
- vocabulary
- logic and reasoning
- data interpretation
- technology literacy

Rules:
- Keep each question self-contained.
- Vary the correct_index across items.
- Make distractors plausible but clearly incorrect.
- Avoid repeated toy questions such as "What is 2 + 2?".
- Do not rely on obscure or time-sensitive facts.
"""

# Backward-compatible alias for older imports. This is the task text, not schema.
EDUCATIONAL_QA_MCQ_TASK = EDU_QA_MCQ_TASK


def build_educational_qa_mcq_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=EDU_QA_MCQ_SCHEMA,
        task_instruction=EDU_QA_MCQ_TASK,
        diversity_context="Use a varied subject, level, and question style.",
    )
