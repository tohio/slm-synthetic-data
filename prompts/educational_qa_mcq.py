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
    "correct_index": { "type": "integer", "minimum": 0, "maximum": 3 },
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
- "correct_index": an integer from 0 to 3, not a string
- "explanation": one short sentence explaining why the correct answer is right

Quality rules:
- Keep each question self-contained.
- Make distractors plausible but clearly incorrect.
- Vary correct_index across items.
- Use the per-batch diversity context as a hard constraint.
- Include concrete details in the question: values, short scenarios, named objects, or a tiny described list/table.
- Avoid generic textbook stems when a scenario can be used instead.
- Avoid repeated toy questions such as "What is 2 + 2?".
- Avoid common trivia questions such as capitals, sky color, largest animal, red planet, or famous authors.
- Do not rely on obscure, controversial, or time-sensitive facts.
- Do not create questions that require current events or private information.

Diversity rules:
- Within one batch, every question must have a different stem pattern.
- Within one batch, every correct answer should be different.
- Within one batch, do not reuse the same four choices.
- Prefer applied questions over pure definitions.
- For arithmetic/math questions, avoid values under 10 unless combined with fractions, decimals, percentages, or a two-step setup.
- For science/history/geography questions, ask about concepts, processes, evidence, or interpretation rather than isolated trivia.

Bad examples to avoid:
- "What is 2 + 2?"
- "What is the capital of France?"
- "What color is the sky?"
- "Which planet is known as the Red Planet?"
- "What is photosynthesis?"

Good example styles:
- "A student has 36 stickers and gives 25% to a friend. How many stickers did the friend get?"
- "A tiny table shows Monday: 8, Tuesday: 12, Wednesday: 10. Which statement best describes the change?"
- "A Python loop stops when x is no longer less than 5. Which condition caused the loop to end?"
- "A plant in a classroom experiment receives less light than another plant. Which explanation best fits the observation?"
"""

# Backward-compatible alias for older imports. This is the task text, not schema.
EDUCATIONAL_QA_MCQ_TASK = EDU_QA_MCQ_TASK


def build_educational_qa_mcq_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=EDU_QA_MCQ_SCHEMA,
        task_instruction=EDU_QA_MCQ_TASK,
        diversity_context="Use a varied subject, level, scenario context, question style, and distractor strategy.",
    )
