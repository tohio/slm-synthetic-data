
FACTUAL_RESTRAINT_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "factual_restraint" },
    "question": { "type": "string" },
    "safe_answer": { "type": "string" }
  },
  "required": ["type", "question", "safe_answer"]
}
"""

FACTUAL_RESTRAINT_TASK = """Generate independent factual restraint records.

Each item must have:
- "type": "factual_restraint"
- "question": a question that tempts hallucination, speculation, or overclaiming
- "safe_answer": a cautious answer that avoids unsupported factual claims

Allowed uncertainty types:
- unknowable future event
- private or unavailable information
- fictional premise presented as real
- overly broad exhaustive request
- ambiguous entity or missing context
- unverifiable statistic
- scientific uncertainty
- proprietary or secret information
- counterfactual scenario
- rumor or unsupported claim

Rules:
- The safe answer should be brief but specific about why the claim cannot be answered reliably.
- Do not invent facts.
- Avoid false certainty.
- Avoid repeating the same generic phrase across records.
- Prefer answers that name the missing evidence or context.
"""


def build_factual_restraint_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=FACTUAL_RESTRAINT_SCHEMA,
        task_instruction=FACTUAL_RESTRAINT_TASK,
        diversity_context="Use a varied uncertainty category and safe-answer style.",
    )
