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

Rules:
- State the input shape, output shape, and required rule in the task.
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

Code rules:
- Produce exactly one function definition and no top-level executable statements.
- Do not import modules, print, include example calls, use classes, raise exceptions, or use try/except.
- Do not use f-strings, format calls, regex, file I/O, or external packages.
- Keep code under 20 lines and faithfully implement the requested transformation and conditions.
"""
