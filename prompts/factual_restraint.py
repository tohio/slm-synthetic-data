
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

FACTUAL_RESTRAINT_TASK = """Generate independent factual-restraint records.

Each item must have:
- "type": "factual_restraint"
- "question": a question that tempts hallucination, overclaiming, or unsupported certainty
- "safe_answer": a cautious answer that avoids unsupported factual claims

Diversity requirements:
- Use the supplied diversity context, batch nonce, uncertainty category, domain, and answer-style hints.
- Do not repeat generic examples such as:
  - "What is the airspeed velocity of an unladen swallow?"
  - "How many rainbows can fit inside a black hole?"
  - "What is the secret ingredient in Coca-Cola?"
  - "How many dimensions are there in the universe?"
  - "How many atoms are in a grain of sand?"
- Do not overuse generic safe answers like "It depends on various factors." or "I don't have enough information."
- Vary the question type across the batch:
  - future prediction
  - private or proprietary information
  - personal/private data
  - unverifiable exact count
  - ambiguous wording
  - missing timeframe or location
  - changing real-world fact
  - speculative science
  - causal claim without evidence
  - precise claim about a fictional or hypothetical world
  - hidden motive/intent claim
  - undocumented historical detail
  - exact ranking or superlative without source
  - medical/legal/financial certainty without context
- Vary domains across the batch:
  - science
  - technology
  - finance
  - history
  - geography
  - medicine
  - law
  - business
  - sports
  - entertainment
  - personal data
  - government/public records
  - fictional or hypothetical scenarios
- The safe answer should be specific to the question.
- The safe answer should briefly explain why the claim cannot be answered reliably.
- When useful, say what information would be needed to answer.
- Keep the safe answer to 1 or 2 sentences.
- Do not invent facts, names, numbers, dates, or citations.
- Do not provide a confident answer to the uncertain question.

Quality requirements:
- The safe answer should be cautious but helpful.
- Avoid repetitive phrasing across items in the same batch.
- Avoid moralizing or unnecessary disclaimers.
- Use plain language.

Additional generation requirements for factual-restraint diversity:
- Avoid generic repeated safe answers such as only "It depends" or "I don't know".
- The safe_answer must be specific to the question and explain what information is missing or why certainty is not appropriate.
- Vary uncertainty categories across the batch: missing context, unverifiable claim, future event, private/secret information, fictional premise, ambiguous wording, disputed estimate, unsupported causal claim, and insufficient data.
- Vary domains across the batch: science, medicine, history, finance, technology, geography, law, sports, entertainment, personal claims, and hypothetical worlds.
- Do not repeat the same question template, same answer sentence, or same uncertainty category within a batch.
- The answer should be cautious but useful; it may state what would be needed to answer reliably.
- Do not fabricate facts, numbers, sources, citations, names, or dates.
"""


def build_factual_restraint_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=FACTUAL_RESTRAINT_SCHEMA,
        task_instruction=FACTUAL_RESTRAINT_TASK,
        diversity_context="Use a varied uncertainty category and safe-answer style.",
    )
