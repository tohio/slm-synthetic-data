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

TASK_CODE_TASK = r"""Generate independent beginner Python programming task records.

Each item must have:
- "type": "task_code"
- "task": a short programming task
- "plan": 2 to 4 short solution steps
- "code": a short Python code snippet solving the task

Allowed task families include:
- list aggregation
- dictionary counting
- set uniqueness
- sorting values
- filtering values
- loops and conditionals
- basic math helper functions
- string cleanup without regular expressions
- grouping items by key
- frequency tables
- nested lists
- simple error handling without file I/O

Disallowed task topics:
- regular expressions
- importing `re`
- CSV parsing
- file I/O
- external packages
- multiline string parsing
- shell commands
- extracting emails, phone numbers, or numeric patterns from text
- tasks that require pattern matching syntax
- custom classes
- custom exceptions

Rules for the "code" field:
- Use Python only.
- Keep code under 20 lines.
- Do not use markdown fences.
- Do not include triple backticks.
- Avoid very long strings.
- Avoid overused tasks such as rectangle area, palindrome, factorial, Fibonacci, and prime checks.
- Use varied function names and varied problem statements.
- The code must be a complete, valid Python 3 snippet.
- The code must pass `ast.parse(code)` with no `SyntaxError` and no `SyntaxWarning`.

Function-only code requirements:
- Return only function definitions and helper logic needed by those functions.
- Do NOT include `print(...)` calls.
- Do NOT include example calls.
- Do NOT include `if __name__ == "__main__":`.
- Do NOT include top-level executable statements.
- Do NOT use f-strings.
- Do NOT generate classes.
- Do NOT raise custom exceptions.
- Prefer one complete function per record.
- A small helper function is allowed only when it is necessary.

JSON and newline rules:
- The response must be valid JSON.
- The `code` field must be a JSON string.
- It is okay for raw JSON text to encode line breaks as `\n`.
- After JSON parsing, the `code` value must contain normal Python line breaks.
- Do not double-escape code so that the parsed `code` value contains literal backslash+n text.
- Do not put Python code in a JSON key.
- Do not output code outside the `code` value.
- Do not leave `code` empty.
- Do not use escaped-code dumps.

Additional strict generation rules for task_code:
- The generated Python code MUST be the value of the `code` field only.
- Do NOT create any extra JSON fields.
- Do NOT create a JSON key whose name contains Python code.
- Every item MUST contain exactly these keys: `type`, `task`, `plan`, `code`.
- `type` MUST be exactly `task_code`.
- `task` MUST be a short non-empty string.
- `plan` MUST be a non-empty list of short strings.
- `code` MUST be a non-empty string containing complete valid Python 3 code.
- Do NOT use markdown code fences.
- Do NOT generate partial code.
- Do NOT generate truncated code.
- Do NOT include prose inside the `code` field.
- Do NOT import `re`.
- Do NOT generate regex tasks.
- Do NOT use regex escapes such as `\d`, `\s`, `\w`, or `\b`.
- Do NOT use invalid string escapes such as `\d`, `\s`, `\w`, `\.`, or `\/`.
- Do NOT use CSV parsing tasks.
- Do NOT use file I/O tasks.
- Do NOT use external packages.
- Do NOT include string literals that contain embedded real newlines.
- If sample input needs multiple values, use Python lists, dictionaries, or simple one-line strings.
- Prefer simple list, dictionary, string, sorting, counting, filtering, grouping, and numeric helper functions.
- Avoid duplicate tasks within a batch.
- The plan must describe the code that is actually returned.

Function definition rules:
- Every function definition must have a complete signature with a closing parenthesis and colon.
- Good: `def summarize_scores(scores):`
- Good: `def combine_inventory(current, incoming):`
- Good: `def select_active_users(users):`
- Bad: `def summarize_scores(scores:`
- Bad: `def combine_inventory(current, incoming:`
- Bad: `def select_active_users(users:`

Task diversity requirements:
- Do not copy task names, function names, sample inputs, or code patterns from these instructions.
- Do not repeatedly generate generic tasks named `count_items`, `remove_duplicates`, `merge_dicts`, or `filter_even_numbers`.
- Vary function names, input variable names, sample values, and data shapes.
- Prefer domain-flavored beginner examples over generic examples.
- Use varied domains such as inventory, grades, tags, usernames, product IDs, scores, categories, logs, labels, short messages, shopping carts, classroom records, and simple metrics.
- For each item in a batch, use a different task family and a different sample input shape.
- Prefer task names that describe a realistic small utility, such as "Summarize product counts", "Select passing grades", or "Group labels by prefix".
- Avoid using the same list values, dictionary keys, or examples repeatedly.
- Avoid code that only wraps a single built-in call unless the task includes a small transformation, validation, or grouping step.

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
  "task": "Summarize product counts",
  "plan": ["Define a function", "Count each product label", "Return the frequency dictionary"],
  "code": "def summarize_products(products):\n    counts = {}\n    for product in products:\n        counts[product] = counts.get(product, 0) + 1\n    return counts"
}
"""


def build_task_code_prompt() -> str:
    from prompts.wrapper import BATCHED_WRAPPER_TEMPLATE

    return BATCHED_WRAPPER_TEMPLATE.format(
        batch_size=1,
        schema=TASK_CODE_SCHEMA,
        task_instruction=TASK_CODE_TASK,
        diversity_context="Use a varied beginner Python topic, implementation pattern, data shape, and domain context. Generate function-only code. Avoid print calls, regex, file I/O, CSV parsing, repeated generic task names, and malformed function signatures.",
    )