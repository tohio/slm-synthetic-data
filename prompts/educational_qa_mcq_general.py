EDUCATIONAL_QA_MCQ_GENERAL_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "educational_qa_mcq_general_candidate"},
        "question": {"type": "string"},
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": ["type", "question", "choices"],
}

EDUCATIONAL_QA_MCQ_GENERAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "integer"},
        "answer": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["candidate_id", "answer", "explanation"],
}

EDUCATIONAL_QA_MCQ_GENERAL_CANDIDATE_TASK = r"""Generate non-math educational multiple-choice candidates without an answer key.

Each item must contain only:
- "type": "educational_qa_mcq_general_candidate"
- "question": one self-contained educational question containing all required evidence
- "choices": exactly four distinct non-empty possible answers

Allowed families:
- Python collection behavior or simple control-flow using literal code in the question
- grammar or vocabulary using supplied text
- reading comprehension using a supplied short passage
- direct fictional-rule application
- technology/privacy policy application using one explicit rule
- scientific-method manipulated-variable identification with one explicit change
- non-numeric categorization or ordering stated in the question

Coherence requirements:
- Generate only a coherent question that can be answered exactly as written using evidence literally included in the question.
- Before returning a candidate, privately determine the one supported answer and confirm that it appears exactly once among the four choices.
- Exactly one choice must be directly supported; the other three choices must be clearly inconsistent with the included evidence.
- If a candidate fails any coherence check below, replace it with a new candidate before returning the batch.

Family-specific coherence checks:
- For Python items, include the complete literal code snippet needed to answer the question; do not refer to a missing snippet.
- For reading, vocabulary, grammar, rule, or policy items, include the exact passage, sentence, rule, or policy needed to answer; do not refer to missing content.
- For fictional-rule or policy application, ask only for a direct consequence explicitly implied by the stated rule.
- For scientific-method items, state the observation or experiment setup and the single deliberately changed variable.
- Do not infer unstated categories, colors, actions, identities, intentions, or real-world facts.
- Do not create a question where multiple choices could reasonably be supported.

Rules:
- Do not include correct_index or explanation.
- Do not generate arithmetic, quantitative calculation, financial calculation, factual recall, ambiguous policy choices, or unsupported inference chains.
- Do not generate questions requiring outside knowledge or common-sense assumptions beyond the stated evidence.
- Provide exactly one supportable answer among the four choices, but do not identify it.
"""

EDUCATIONAL_QA_MCQ_GENERAL_RESPONSE_TASK = r"""Independently answer each general educational MCQ candidate.

For each candidate_id, return only the answer-side fields:
- "candidate_id": the supplied id
- "answer": the one answer supported by the supplied evidence; copy the matching candidate choice exactly when it is already usable
- "explanation": one concise sentence citing the supplied code, passage, sentence, rule, or experimental setup

Answer-verification requirements:
- Determine the answer using only evidence literally supplied in the candidate question.
- Before returning each item, confirm that "answer" is the one direct conclusion established by that evidence and that "explanation" supports exactly that answer.
- If the question refers to a missing passage, rule, policy, sentence, experiment setup, or code snippet, return empty strings for "answer" and "explanation" so the item is rejected locally.
- If more than one choice is defensible, or no answer follows directly from the stated evidence, return empty strings for "answer" and "explanation" so the item is rejected locally.
- If answering would require an outside fact, unstated assumption, quantitative calculation, or financial calculation, return empty strings for "answer" and "explanation" so the item is rejected locally.

Rules:
- If one candidate choice exactly expresses the supported answer, copy that choice text exactly into "answer".
- If the candidate is otherwise coherent but none of its choices states the supported answer clearly, return one concise replacement answer; Python will replace one distractor.
- Do not return question, choices, or correct_index. Python assembles those fields from your answer.
- Do not turn the item into a math question.
- Do not mention corrections, errors, answer keys, or the generation process.
"""
