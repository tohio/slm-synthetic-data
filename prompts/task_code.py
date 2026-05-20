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

Allowed task topics include:
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
- Good: `def count_items(items):`
- Good: `def merge_dicts(a, b):`
- Good: `def filter_even_numbers(numbers):`
- Bad: `def count_items(items:`
- Bad: `def merge_dicts(a, b:`
- Bad: `def filter_even_numbers(numbers:`

Preferred task examples:
- Count items in a list.
- Count word frequencies using `text.split()`.
- Remove duplicates from a list.
- Sort numbers or strings.
- Filter even numbers.
- Find maximum or minimum values.
- Merge two dictionaries.
- Invert a dictionary.
- Group strings by first letter.
- Count character frequencies.
- Compute an average from a list of numbers.
- Flatten a nested list.
- Check whether all numbers are positive.

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
        diversity_context="Use a varied beginner Python topic, implementation pattern, and constraint. Avoid regex, file I/O, CSV parsing, and malformed function signatures.",
    )