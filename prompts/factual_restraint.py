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

The safe answer should be brief, should not invent facts, and should avoid false certainty.
"""


def build_factual_restraint_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=FACTUAL_RESTRAINT_SCHEMA,
        task_instruction=FACTUAL_RESTRAINT_TASK,
    )
