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
- "question": one clear, self-contained educational question that includes all required evidence
- "choices": exactly 4 distinct non-empty answer choices
- "correct_index": the actual position of the one supported answer after the choices are written
- "explanation": one short sentence explaining why choices[correct_index] follows from the supplied evidence

Allowed subject families:
- Python collection behavior using a literal short code expression included in the question
- Python control-flow interpretation using a literal short `if`, `break`, or `continue` snippet included in the question
- grammar and sentence structure using one supplied sentence
- vocabulary in context using one supplied sentence and one target word
- reading comprehension using a supplied two-sentence passage
- direct rule application using fictional objects and an explicit if-then rule
- technology and privacy literacy using one explicit security rule stated in the question
- scientific method asking for the deliberately changed variable in an experiment setup that names that one change
- non-numeric categorization or ordering using labels or an order explicitly stated in the question

Do NOT generate:
- arithmetic, fractions, percentages, ratios, algebra, geometry, probability calculations, statistics calculations, measurement calculations, or financial calculations
- a question that asks for numeric computation or has numeric-answer choices
- questions requiring external factual recall, current knowledge, real-world biological facts, or obscure domain facts
- history/date recall or geography/location recall
- open-ended "what should happen next" prompts
- subjective choices such as purchases, personal decisions, or the "best" action without an explicit rule
- logic questions using "all/some" quantifier chains, converse reasoning, or unstated assumptions
- Python questions that refer to a code snippet without including the literal snippet
- scientific-method questions about controlled variables when more than one control may be valid
- security questions where more than one option could reasonably be safe
- questions whose answer depends on information not stated in the question

Family-specific rules:
- Python: include the literal code being interpreted, and make the answer describe that exact code.
- Rule/logic: use fictional labels or objects; ask only for a direct consequence of a stated rule.
- Security/privacy: state the policy first; make only one choice comply exactly with that policy.
- Scientific method: state the single deliberately changed variable; ask for that manipulated variable only.
- Reading/vocabulary/grammar: the answer must be directly supported by the supplied text.

Answer construction rules:
1. Write the supporting evidence or rule in the question.
2. Determine the one answer supported by that evidence.
3. Write three choices that are contradicted by, or unsupported by, the evidence.
4. Place the supported answer once among the four choices.
5. Set correct_index to the position where the supported answer actually appears.
6. Write an explanation that names or restates choices[correct_index] and the supporting evidence.
7. Before returning, verify that the explanation supports choices[correct_index] and no other choice.

Quality rules:
- Do not target a predetermined answer position; correctness is more important than balanced option locations.
- Keep answer choices distinct and concise.
- Use the per-batch non-math diversity profile as a hard constraint.
- Do not include any extra top-level keys.
"""


def build_educational_qa_mcq_general_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE

    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=EDUCATIONAL_QA_MCQ_GENERAL_SCHEMA,
        task_instruction=EDUCATIONAL_QA_MCQ_GENERAL_TASK,
        diversity_context=(
            "Generate one self-contained non-math educational MCQ. "
            "Determine the supported answer before choosing its position, "
            "then verify that the explanation supports the indexed choice."
        ),
    )
