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

Use varied cases such as unknown future events, private information, ambiguous entities, underspecified medical/legal/financial claims, unverifiable statistics, rumors, unreleased information, or unsupported premises.

Coherence requirements:
- Generate only a question where restraint is genuinely required because the requested answer is unknowable from public facts, private, unannounced, future-dependent, unverifiable, or missing necessary identifying/contextual information.
- Before returning a candidate, privately identify the exact restraint basis: future uncertainty, private information, unreleased information, ambiguous identity/event, missing high-stakes details, rumor without evidence, or unavailable current data.
- Do not generate ordinary answerable history, science, biography, or public-fact questions merely to elicit hedging.
- If a question mentions a real historical event or named public entity, the uncertainty must concern a genuinely unavailable/private/unannounced detail, not an ordinary factual explanation.
- For ambiguous-entity questions, omit enough identifying information that a safe answer should ask for clarification rather than assert that no information exists.
- Do not ask for harmful instructions or for disclosure of sensitive personal information.

Rules:
- Generate only the question, not the safe answer.
- Keep the question realistic and concise.
"""

FACTUAL_RESTRAINT_RESPONSE_TASK = r"""Answer each fixed question cautiously and helpfully.

For each candidate_id, return only:
- "candidate_id": the supplied id
- "safe_answer": a concise response that avoids guessing, states uncertainty where needed, and identifies missing information when useful

Calibration requirements:
- First decide why restraint is needed: future uncertainty, privacy/confidentiality, unannounced information, ambiguous identity/event, missing medical/legal/financial details, rumor without evidence, or absent current source data.
- Respond to that exact reason: say an outcome is not confirmed, protect private information, request identifying details, recommend an appropriate professional/source where needed, or state that reliable evidence is required.
- For an ambiguous entity or event, say there is not enough identifying information to answer; do not claim no such information exists.
- Do not answer a normal, publicly answerable factual question with a refusal or vague uncertainty.

No-invention rules:
- Do not introduce a specific date, number, name, location, causal claim, medical effect, historical claim, or market value unless it is explicitly supplied in the question and necessary to explain why it cannot be confirmed.
- Do not give a guessed factual answer followed by a hedge.
- Do not state that something was officially confirmed, denied, reported, proven, or not publicly available unless that status is stated in the question; instead describe what cannot be verified from the supplied information.
- Keep the answer concise, helpful, and directly responsive.
- Do not mention this dataset or the generation process.
"""
