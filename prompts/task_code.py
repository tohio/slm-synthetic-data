TASK_CODE_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "task_code_candidate"},
        "task": {"type": "string"},
    },
    "required": ["type", "task"],
}

TASK_CODE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "integer"},
        "plan": {"type": "array", "items": {"type": "string"}},
        "code": {"type": "string"},
    },
    "required": ["candidate_id", "plan", "code"],
}

TASK_CODE_CANDIDATE_TASK = r"""Generate independent beginner/intermediate Python programming tasks without solutions.

Each item must contain only:
- "type": "task_code_candidate"
- "task": one short, complete task specification that can be solved by exactly one Python function

Allowed task families include filtering plus aggregation, grouping, transformed totals, comparisons, sorting structured values, normalized token counting, and nested-list transformations.

Coherence requirements:
- Generate exactly one internally consistent function task with one clear input contract and one clear output contract.
- Before returning the task, privately outline a solution under 20 lines and confirm every stated constraint is necessary and implementable together.
- Keep each task focused: use at most one transformation, one filter condition, one grouping or selection operation, and one ordering requirement unless the operations naturally form a single pipeline.
- Do not add unrelated cutoffs, divisors, status labels, derived keys, ordering directions, or secondary constants merely for variety.
- Do not require a full-token key and a prefix-derived key for the same output dictionary.
- Do not combine numeric sequence comparison with text normalization, status labels, or unrelated bucket rules.
- State whether inputs may be mutated; default to preserving inputs.
- If any instruction conflicts with another, is unused by the intended function, or cannot be implemented in the requested output shape, replace the candidate before returning the batch.

Rules:
- State the input shape, output shape, and every required transformation/filter/grouping/order rule in the task.
- The intended solution must use one function only, under 20 lines, with no imports or external packages.
- Do not request regex, file I/O, CSV parsing, classes, exceptions, shell commands, printing, or example calls.
- Do not provide a plan, code, solution hint, or completed function.
- Avoid trivial repeated beginner tasks and familiar copied function names.
"""

TASK_CODE_RESPONSE_TASK = r"""Solve each fixed Python task independently.

For each candidate_id, return only:
- "candidate_id": the supplied id
- "plan": 2 to 4 short implementation steps
- "code": one complete valid Python 3 function definition solving the task

Adherence requirements:
- Break the task into its atomic requirements before writing code: input shape, output shape, transformation, filter, grouping/selection, ordering, mutation rule, and any label/status rule.
- Return code only when every stated requirement can be implemented together exactly as written.
- Confirm before returning that the function implements every atomic requirement and does not add, omit, or reinterpret any condition.
- Preserve input lists and dictionaries unless the task explicitly requires mutation; do not call in-place sorting methods on inputs.
- If the task contains contradictory output keys, unrelated requirements, impossible conditions, or requirements that would be ignored, return an empty plan array and an empty code string so the item is rejected locally.

Code rules:
- Produce exactly one function definition and no top-level executable statements.
- Do not import modules, print, include example calls, use classes, raise exceptions, or use try/except.
- Do not use f-strings, format calls, regex, file I/O, or external packages.
- Keep code under 20 lines and faithfully implement every requested transformation and condition.
"""
