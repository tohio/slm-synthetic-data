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

Rules:
- Do not include correct_index or explanation.
- Do not generate arithmetic, quantitative calculation, financial calculation, factual recall, ambiguous policy choices, or unsupported inference chains.
- Provide exactly one supportable answer among the four choices, but do not identify it.
"""

EDUCATIONAL_QA_MCQ_GENERAL_RESPONSE_TASK = r"""Independently answer each general educational MCQ candidate.

For each candidate_id, return only the answer-side fields:
- "candidate_id": the supplied id
- "answer": the one answer supported by the supplied evidence; copy the matching candidate choice exactly when it is already usable
- "explanation": one concise sentence citing the supplied code, passage, sentence, rule, or experimental setup

Rules:
- Determine the answer from supplied evidence only.
- Do not return question, choices, or correct_index. Python assembles those fields from your answer.
- If none of the candidate choices expresses the supported answer, return the correct answer text; Python will replace one distractor.
- Do not turn the item into a math question.
- Do not mention corrections, errors, answer keys, or the generation process.
"""
