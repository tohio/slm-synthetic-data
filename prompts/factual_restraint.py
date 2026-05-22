FACTUAL_RESTRAINT_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "factual_restraint_candidate"},
        "question": {"type": "string"},
    },
    "required": ["type", "question"],
}

FACTUAL_RESTRAINT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "integer"},
        "safe_answer": {"type": "string"},
    },
    "required": ["candidate_id", "safe_answer"],
}

FACTUAL_RESTRAINT_CANDIDATE_TASK = r"""Generate questions that should be answered with factual restraint.

Each item must contain only:
- "type": "factual_restraint_candidate"
- "question": one concise question where an assistant must avoid inventing unsupported information

Use varied cases such as unknown future events, private information, ambiguous entities, underspecified medical/legal/financial claims, unverifiable statistics, rumors, or unsupported premises.

Rules:
- Generate only the question, not the safe answer.
- Keep the question realistic and concise.
- Do not require unsafe instructions or sensitive personal disclosure.
"""

FACTUAL_RESTRAINT_RESPONSE_TASK = r"""Answer each fixed question cautiously and helpfully.

For each candidate_id, return only:
- "candidate_id": the supplied id
- "safe_answer": a concise response that avoids guessing, states uncertainty where needed, and identifies missing information when useful

Rules:
- Do not invent facts, predictions, private information, or definitive high-stakes conclusions without supplied evidence.
- Do not mention this dataset or the generation process.
"""
