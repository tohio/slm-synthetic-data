EDUCATIONAL_QA_MCQ_GENERAL_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "educational_qa_mcq_general" },
    "question": { "type": "string" },
    "choices": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 4,
      "maxItems": 4
    },
    "correct_index": { "type": "integer", "minimum": 0, "maximum": 3 },
    "explanation": { "type": "string" }
  },
  "required": ["type", "question", "choices", "correct_index", "explanation"]
}
"""

EDUCATIONAL_QA_MCQ_GENERAL_TASK = r"""Generate independent non-math educational multiple-choice records.

Each item must have exactly these fields:
- "type": "educational_qa_mcq_general"
- "question": one clear, self-contained educational question
- "choices": exactly 4 distinct non-empty answer choices
- "correct_index": an integer from 0 to 3 pointing to the correct answer
- "explanation": one short sentence explaining why the indexed answer follows from the supplied information

Allowed subject families:
- Python data types, collection behavior, and simple control-flow interpretation using a short code fragment stated in the question
- computer-science concepts illustrated by a concrete, self-contained scenario
- grammar and sentence structure using a sentence supplied in the question
- vocabulary in context using a sentence supplied in the question
- reading comprehension using a short passage supplied in the question
- logic and inference using only facts supplied in the question
- technology and privacy literacy using stable safety practices
- scientific method using a described observation or experiment setup
- non-numeric data interpretation using labels, categories, or ordering stated in the question

Do NOT generate:
- arithmetic, fractions, percentages, ratios, algebra, geometry, probability calculations, statistics calculations, measurement calculations, or financial calculations
- questions requiring external factual recall, current knowledge, or obscure domain facts
- history/date recall or geography/location recall
- open-ended "what should happen next" prompts
- subjective choices such as purchases, personal decisions, or the "best" action without an explicit rule
- questions whose answer depends on information not stated in the question

Quality rules:
- The correct answer must be supported directly by a supplied sentence, passage, code fragment, rule, observation, or scenario detail.
- Make the distractors plausible but clearly contradicted by the supplied information.
- Keep answer choices distinct and concise.
- Vary correct_index across records.
- Use the per-batch non-math diversity profile as a hard constraint.
- Do not include any extra top-level keys.
"""


def build_educational_qa_mcq_general_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE

    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=EDUCATIONAL_QA_MCQ_GENERAL_SCHEMA,
        task_instruction=EDUCATIONAL_QA_MCQ_GENERAL_TASK,
        diversity_context="Generate one self-contained non-math educational MCQ.",
    )
