EDUCATIONAL_QA_MCQ_TASK = r"""
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

EDU_QA_MCQ_TASK = """Generate a single multiple-choice educational question.

The JSON object MUST have:
- "type": "educational_qa_mcq"
- "question": the question text
- "choices": an array of exactly 4 answer options (strings)
- "correct_index": an integer index 0–3 pointing into "choices"
- "explanation": a short explanation of why the correct choice is correct
"""

def build_educational_qa_mcq_prompt() -> str:
    return WRAPPER_TEMPLATE.format(
        schema=EDU_QA_MCQ_SCHEMA,
        task_instruction=EDU_QA_MCQ_TASK,
    )
