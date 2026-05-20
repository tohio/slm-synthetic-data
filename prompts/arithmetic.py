ARITHMETIC_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "arithmetic" },
    "question": { "type": "string" },
    "answer": { "type": "string" },
    "steps": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["type", "question", "answer", "steps"]
}
"""

ARITHMETIC_TASK = r"""Generate independent arithmetic reasoning records.

Each item must have:
- "type": "arithmetic"
- "question": a short integer arithmetic question
- "steps": 2 to 4 short strings explaining the calculation
- "answer": the final answer as a string

Core requirements:
- Keep arithmetic LLM-generated.
- Use only integer arithmetic.
- The calculation must be correct.
- The answer must match the final step.
- The answer must be only the final numeric result as a string.
- Keep all questions self-contained.
- Use concise natural language.
- Steps should explain the calculation, not just restate the question.

Allowed arithmetic forms:
- direct equation
- short word problem
- missing operand
- compare two expressions
- order three computed values
- two-step arithmetic
- rate or quantity word problem
- inventory or accounting-style word problem
- distance, time, or unit-count word problem
- grouped multiplication plus addition
- subtraction after grouping
- equal sharing with exact integer division
- simple budget, ticket, package, classroom, warehouse, or schedule scenario

Operation families:
- addition
- subtraction
- multiplication
- exact integer division
- mixed addition and subtraction
- multiplication followed by addition
- multiplication followed by subtraction
- division followed by addition
- comparison of two computed values
- ordering of three computed values

Number range requirements:
- Use varied integer ranges.
- Use some small values from 1 to 99.
- Use some medium values from 100 to 9,999.
- Use some larger values from 10,000 to 999,999 when the calculation is still easy to verify.
- Include negative results only in clear subtraction or comparison questions.
- Use exact integer division only; choose numbers so the answer has no remainder.
- Do not use decimals, fractions, percentages, or remainders.
- Do not use algebra beyond a single missing integer.

Diversity requirements:
- Avoid template collapse across a long run.
- Do not rely on the same small set of operands.
- Do not repeatedly use the same number pair, same operation sequence, same answer, or same question stem.
- Do not repeatedly use direct equations of the form "A + B = ?", "A - B = ?", "A × B = ?", or "A ÷ B = ?".
- Direct equations are allowed, but prefer varied wording and uncommon operands.
- Prefer concrete, varied contexts when using word problems.
- Use different nouns, entities, units, and scenarios across records.
- Vary step wording across records.
- Vary whether the question asks for a total, remaining amount, missing amount, larger value, smaller value, difference, product, quotient, or ordered result.
- Avoid using the same operation with the same context repeatedly.

Disallowed repetitive examples and stems:
- Do not generate "47 + 29 = ?".
- Do not generate "47 - 29 = ?".
- Do not generate repeated 47-and-29 arithmetic patterns.
- Do not generate repeated bookshelf/shelf/book capacity problems.
- Do not generate repeated Sally/Tom pencil giveaway problems.
- Do not generate repeated basket/apple eaten problems.
- Do not generate repeated bakery loaves-per-day problems.
- Do not generate generic toy examples such as "2 + 2 = ?", "2 + 5 = ?", "What is 2 + 2?", or repeated "Add X and Y" prompts.
- Do not repeatedly ask "What is A plus B?", "What is A minus B?", or "A + B = ?".
- Do not use the exact phrases "The result is" or "Add X and Y" as the dominant step style.

Preferred context domains:
- warehouse inventory
- classroom supplies
- train or bus seats
- event tickets
- meal boxes
- package shipments
- library checkouts
- parking spaces
- game scores
- exercise counts
- warehouse shelves without using the common bookshelf template
- store receipts
- simple budgets
- production batches
- delivery routes
- time blocks
- resource planning
- team assignments
- crates, cartons, bins, pallets, or boxes
- pages, labels, badges, forms, or cards

Question style requirements:
- For missing-operand items, make the missing value clear and ensure the answer is the missing operand.
- For comparison items, ask which computed value is larger or smaller.
- For ordering items, ask for the ordered numeric results or the largest/smallest computed result.
- For two-step items, keep the arithmetic simple enough to verify but make the wording distinct.
- For word problems, use a realistic noun and avoid repeating common toy names.
- For direct equations, use less common operands and vary the operator placement.

Step requirements:
- Use 2 to 4 short steps.
- Each step must be faithful to the arithmetic.
- Do not include long explanations.
- Do not include irrelevant story details.
- Do not merely copy the question into the steps.
- The final step should make the final numeric answer clear.
- Do not use chain-of-thought style prose; keep steps compact and calculation-focused.

Quality requirements:
- The problem should be clear without external context.
- The calculation must be exactly correct.
- The answer must be an exact integer string.
- Do not include fractions, decimals, percentages, or ambiguous wording.
"""


def build_arithmetic_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE

    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=ARITHMETIC_SCHEMA,
        task_instruction=ARITHMETIC_TASK,
        diversity_context=(
            "Use a varied arithmetic form, uncommon operands, operation family, "
            "number range, and concrete context. Avoid repeated direct-equation "
            "templates, repeated toy word problems, and repeated number pairs."
        ),
    )