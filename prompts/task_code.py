
TASK_CODE_SCHEMA = r"""
{
  "type": "object",
  "properties": {
    "type": { "const": "task_code" },
    "task": { "type": "string" },
    "plan": {
      "type": "array",
      "items": { "type": "string" }
    },
    "code": { "type": "string" }
  },
  "required": ["type", "task", "plan", "code"]
}
"""

TASK_CODE_TASK = """Generate independent beginner Python programming task records.

Each item must have:
- "type": "task_code"
- "task": a short programming task
- "plan": 2 to 4 short solution steps
- "code": a short Python code snippet solving the task

Allowed task topics include:
- strings and text cleanup
- lists and simple aggregation
- dictionaries and counting
- sets and uniqueness
- sorting records
- filtering values
- loops and conditionals
- basic math helper functions
- parsing simple delimited strings
- input validation helpers
- nested lists
- grouping items by key
- frequency tables
- simple error handling

Rules for "code":
- Use Python only.
- Keep code under 20 lines.
- Do not use markdown fences.
- Do not include triple backticks.
- Avoid very long strings.
- JSON must escape newline characters correctly inside the code string.
- Avoid overused tasks such as rectangle area, palindrome, factorial, Fibonacci, and prime checks.
- Use varied function names and varied problem statements.

Additional generation requirements for clean Python code:
- Produce valid Python 3 code that passes ast.parse without SyntaxError or SyntaxWarning.
- Use real newline characters in the code string content; do not output literal escaped "\\n" or "\\t" sequences as formatting.
- If regular expressions are needed, use raw strings such as r"\\d+", r"\\s+", or r"[A-Za-z]+".
- Do not use invalid string escapes such as "\\d", "\\s", "\\w", or "\\." inside normal quoted strings.
- Do not generate partial code, truncated code, markdown fences, shell commands, or prose inside the code field.
- Prefer small self-contained functions with clear names, simple inputs, and optional tiny examples.
- Avoid duplicate tasks within a batch. Vary task domains: strings, lists, dictionaries, files-as-strings, parsing, validation, counting, sorting, filtering, simple math, dates-as-strings, and CSV-like text.
- The plan must describe the code that is actually returned.

Additional strict generation rules for task_code:
- The generated Python code MUST be the value of the `code` field only.
- Do NOT create any extra JSON fields.
- Do NOT create a JSON key whose name contains Python code.
- Every item MUST contain exactly these keys: `type`, `task`, `plan`, `code`.
- `type` MUST be exactly `task_code`.
- `task` MUST be a short non-empty string.
- `plan` MUST be a non-empty list of short strings.
- `code` MUST be a non-empty string containing complete valid Python 3 code.
- The `code` string MUST contain real line breaks inside the JSON string value after parsing.
- Do NOT output code outside the `code` value.
- Do NOT use markdown code fences.
- Do NOT use escaped-code dumps where code appears as a JSON key.
- Do NOT leave `code` empty.
- Avoid regular-expression tasks for now.
- Avoid CSV parsing tasks that require embedded multi-line string literals.
- Avoid file I/O tasks for now.
- Avoid tasks requiring external packages.
- Prefer simple list, dictionary, string, sorting, counting, filtering, grouping, and numeric helper functions.
- If sample input needs multiple values, use Python lists, dictionaries, or simple one-line strings.
- Do not include string literals that contain embedded real newlines, such as `"line1
line2"`; use a list of strings instead.
- Do not use regex escapes such as `\d`, `\s`, `\w`, or `\b`.
- The code must pass `ast.parse(code)` with no `SyntaxError` and no `SyntaxWarning`.

Bad output pattern to avoid:
{
  "type": "task_code",
  "task": "Count words",
  "plan": ["Split text", "Count words"],
  "code": "",
  "def count_words(text):\n    return len(text.split())": ""
}

Correct output pattern:
{
  "type": "task_code",
  "task": "Count words in a sentence",
  "plan": ["Define a function", "Split the sentence", "Return the number of words"],
  "code": "def count_words(text):\n    return len(text.split())\n\nprint(count_words('hello world'))"
}
"""


def build_task_code_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE
    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=TASK_CODE_SCHEMA,
        task_instruction=TASK_CODE_TASK,
        diversity_context="Use a varied programming topic, implementation pattern, and constraint.",
    )
