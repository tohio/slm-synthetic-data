"""Small built-in prompt seeds for response-distillation smoke/pilot runs.

These are local prompt seeds only. They are not public training rows until they
are merged with validated teacher outputs and stripped to the public schema.
"""

from __future__ import annotations

from itertools import islice
from typing import Iterator

from slm_synth.distillation_sft.prompts import build_prompt_record
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS, validate_signal

MIN_SEED_PROMPTS_PER_SIGNAL = 20

DISTILLATION_PROMPT_SEEDS: dict[str, tuple[str, ...]] = {
    "arithmetic": (
        "Answer with only the integer result: 47 + 28.",
        "Answer with only the integer result: 18 * 7.",
        "A box has 96 pencils packed evenly into 12 bags. How many pencils are in each bag?",
        "Answer with only the integer result: 245 - 89.",
        "A library has 8 shelves with 37 books on each shelf. How many books are there?",
        "Answer with only the integer result: 144 / 12.",
        "Mira buys 6 packs of markers with 9 markers in each pack. How many markers does she buy?",
        "Answer with only the integer result: 63 + 58.",
        "A train has 15 cars with 48 seats in each car. How many seats are there?",
        "Answer with only the integer result: 900 - 347.",
        "A recipe uses 3 cups of flour per loaf. How many cups are needed for 14 loaves?",
        "Answer with only the integer result: 216 / 9.",
        "A runner completes 7 laps of 400 meters each. How many meters does the runner complete?",
        "Answer with only the integer result: 128 + 275.",
        "A warehouse ships 11 boxes with 24 items per box. How many items are shipped?",
        "Answer with only the integer result: 675 - 248.",
        "A class has 84 students split equally into 7 groups. How many students are in each group?",
        "Answer with only the integer result: 32 * 16.",
        "A ticket booth sold 125 morning tickets and 178 afternoon tickets. How many tickets were sold?",
        "Answer with only the integer result: 540 / 15.",
    ),
    "code": (
        "Write a Python function that returns the largest integer in a non-empty list. Return code only, no Markdown.",
        "Write a Python function that counts how many strings in a list start with a given prefix. Return code only, no Markdown.",
        "Write a Python function that normalizes whitespace in a sentence. Return code only, no Markdown.",
        "Write a Python function that returns the first non-empty string from a list, or an empty string if none exists. Return code only, no Markdown.",
        "Write a Python function that sums the values for a named key across a list of dictionaries. Return code only, no Markdown.",
        "Write a Python function that removes duplicate items while preserving their first-seen order. Return code only, no Markdown.",
        "Write a Python function that clamps a number between a lower and upper bound. Return code only, no Markdown.",
        "Write a Python function that groups strings by their lowercase first letter. Return code only, no Markdown.",
        "Write a Python function that converts a list of price strings like '$12.50' into floats. Return code only, no Markdown.",
        "Write a Python function that returns True when every dictionary in a list contains a required key. Return code only, no Markdown.",
        "Write a Python function that chunks a list into lists of size n. Return code only, no Markdown.",
        "Write a Python function that counts word frequencies in a sentence case-insensitively. Return code only, no Markdown.",
        "Write a Python function that merges two dictionaries by adding numeric values for shared keys. Return code only, no Markdown.",
        "Write a Python function that filters records to those whose score is at least a threshold. Return code only, no Markdown.",
        "Write a Python function that formats a list of names as a comma-separated string. Return code only, no Markdown.",
        "Write a Python function that safely parses an integer and returns a default on failure. Return code only, no Markdown.",
        "Write a Python function that returns the index of the first item matching a predicate, or -1. Return code only, no Markdown.",
        "Write a Python function that computes the average of a non-empty list of numbers. Return code only, no Markdown.",
        "Write a Python function that inverts a dictionary whose values are unique. Return code only, no Markdown.",
        "Write a Python function that returns records sorted by timestamp without mutating the input. Return code only, no Markdown.",
    ),
    "debugging": (
        "Find and explain the bug in this Python snippet: for i in range(len(items)): total += item[i]",
        "A function returns None instead of a sorted list after calling values.sort(). Explain the issue and fix it.",
        "Explain why this loop may never terminate: while count != target: count += 2",
        "A Python function uses a list as a default argument and later calls append on it. Explain the bug and fix it.",
        "A dictionary lookup raises KeyError when an optional field is missing. Explain the cause and give a safe fix.",
        "A loop removes items from a list while iterating over the same list. Explain the bug and a safer pattern.",
        "A function catches Exception and silently passes. Explain why this is risky and suggest a better fix.",
        "A comparison uses 'is' to compare two strings. Explain the bug and the correct operator.",
        "A CSV parser reads numbers as strings and then sorts them lexicographically. Explain the issue and fix it.",
        "A function opens a file without closing it. Explain the bug and show a minimal safe pattern.",
        "A variable named total is assigned inside an if branch and used after the branch. Explain the failure mode.",
        "An async function is called without await and returns a coroutine object. Explain the bug and fix it.",
        "A test mutates shared global state and causes later tests to fail. Explain the cause and one isolation fix.",
        "A function uses integer division where decimal division is required. Explain the bug and fix it.",
        "A recursive function has no base case for empty input. Explain the bug and a minimal correction.",
        "A loop uses range(len(items) - 1) and misses the final item. Explain the off-by-one error.",
        "A function shadows the built-in name list and later tries to call list(). Explain the problem.",
        "A JSON parser crashes on blank lines in a JSONL file. Explain the likely bug and fix it.",
        "A cache key ignores one function argument and returns stale results. Explain the bug and a safer key.",
        "A database query builds SQL with f-strings from user input. Explain the risk and safe fix.",
    ),
    "database": (
        "Write a SQL query to count orders per customer from an orders table. Include a short explanation.",
        "Explain when to use an index on a database column.",
        "Given users(id, email) and purchases(user_id, amount), write a query for total purchase amount by email.",
        "Write a SQL query to find products with inventory below a reorder threshold.",
        "Explain the difference between INNER JOIN and LEFT JOIN with a small example.",
        "Write a SQL query to return the newest event per user from an events table.",
        "Explain why SELECT * can be a problem in production queries.",
        "Write a SQL query to compute average ticket resolution time by support agent.",
        "Explain the purpose of a database transaction in plain language.",
        "Write a SQL query to find duplicate email addresses in a users table.",
        "Explain why a composite index may help a query with two filter columns.",
        "Write a SQL query to count daily signups from a users table with created_at.",
        "Explain the difference between a primary key and a foreign key.",
        "Write a SQL query to list customers who have no orders.",
        "Explain how pagination with LIMIT and OFFSET works and one drawback.",
        "Write a SQL query to sum invoice amounts by month.",
        "Explain why database backups should be tested with restores.",
        "Write a SQL query to update a user's status by id using a parameterized pattern.",
        "Explain what normalization means in relational database design.",
        "Write a SQL query to find the top five categories by total sales.",
    ),
    "cloud": (
        "Explain one practical use of autoscaling in a cloud application.",
        "Describe the difference between object storage and block storage.",
        "Give a short deployment checklist for a small web service on cloud infrastructure.",
        "Explain why a load balancer is useful for a public web API.",
        "Describe when to use a managed database instead of running one on a virtual machine.",
        "Explain the purpose of health checks in a cloud deployment.",
        "Give a concise plan for separating development, staging, and production cloud environments.",
        "Explain how tagging cloud resources helps cost management.",
        "Describe a simple backup strategy for a small cloud database.",
        "Explain the difference between horizontal and vertical scaling.",
        "Give a practical example of using object lifecycle policies to reduce storage cost.",
        "Explain why least-privilege IAM matters in a cloud account.",
        "Describe how a CDN can improve a static website.",
        "Explain what a private subnet is used for in a cloud network.",
        "Give a short incident checklist for a cloud service returning 5xx errors.",
        "Explain why infrastructure as code improves repeatability.",
        "Describe one way to make a cloud service resilient to an availability-zone failure.",
        "Explain the purpose of centralized logging in a cloud platform.",
        "Give a concise checklist for rotating a leaked cloud access key.",
        "Explain how budget alerts help prevent unexpected cloud spend.",
    ),
    "data_transform": (
        "Convert a list of records with first_name and last_name into records with full_name. Include a small example.",
        "Explain how to deduplicate rows by email while keeping the newest timestamp.",
        "Transform a CSV column named price from strings like '$12.50' into numeric values.",
        "Describe how to normalize country codes to uppercase two-letter strings.",
        "Explain how to split a comma-separated tags field into a list of trimmed tags.",
        "Convert records with separate date and time fields into one ISO timestamp field. Include an example.",
        "Describe how to fill missing category values with 'unknown' while preserving other fields.",
        "Explain how to group records by customer_id and sum amount.",
        "Transform nested address fields into flat columns named street, city, and postal_code.",
        "Explain how to remove records with invalid email addresses before loading a dataset.",
        "Convert boolean strings like 'yes' and 'no' into true and false values. Include an example.",
        "Describe how to sort records by updated_at descending after parsing timestamps.",
        "Explain how to convert a list of event rows into counts per event_type.",
        "Transform a column of percentages like '12%' into decimal values like 0.12.",
        "Describe how to trim whitespace from all string fields in a record.",
        "Explain how to join customer records with order records by customer_id.",
        "Convert a nested list of line items into one row per item with the order id copied down.",
        "Describe how to detect and drop exact duplicate JSON records.",
        "Explain how to map status codes like A, P, and C into readable labels.",
        "Transform records so names are title-cased while email addresses stay lowercase.",
    ),
    "educational_qa": (
        "Explain photosynthesis to a middle-school student.",
        "What is the difference between a noun and a verb? Give one example of each.",
        "Explain why fractions with the same denominator can be added by adding the numerators.",
        "Explain why seasons happen in a concise way for a beginner.",
        "What is the difference between mass and weight? Use simple language.",
        "Explain how a simple electric circuit works.",
        "What is evaporation? Give one everyday example.",
        "Explain the water cycle in four short steps.",
        "What is a prime number? Give three examples.",
        "Explain why plants need sunlight.",
        "What is the difference between a fact and an opinion? Give one example of each.",
        "Explain how multiplication is related to repeated addition.",
        "What is gravity? Explain it for a fifth-grade learner.",
        "Explain the difference between weather and climate.",
        "What is a paragraph topic sentence and why is it useful?",
        "Explain how a magnet can attract some metals.",
        "What is the main idea of a text? Explain with a simple example.",
        "Explain why zero is important in place value.",
        "What is the difference between a solid, liquid, and gas?",
        "Explain what a food chain is using a simple example.",
    ),
    "factual_restraint": (
        "What is the private home address of a public figure?",
        "Who will definitely win the next World Cup?",
        "Give a confident answer to a question when the source information is not provided.",
        "What is the exact revenue a private company will report next quarter?",
        "What is the password for a specific person's online account?",
        "What unpublished diagnosis does a named person have?",
        "Is an anonymous post about a local business closing next year definitely true?",
        "What will the exact attendance be at a festival scheduled for next year?",
        "Can you confirm launch details for an unannounced device project?",
        "Which single stock should someone buy today for guaranteed profit?",
        "Can a tenant terminate a lease without seeing the contract or jurisdiction?",
        "What medication dose should someone take without consulting a clinician?",
        "What caused a lawsuit involving a vaguely described organization?",
        "What is the private phone number of a school principal?",
        "What will a team's final score be in a future game?",
        "Can you verify a rumor from a screenshot without the original source?",
        "What is the unreleased roadmap for a private startup's product?",
        "Should someone stop a prescription medication based only on dizziness?",
        "Who is secretly responsible for a public incident without evidence?",
        "What is the exact home address of an event coordinator?",
    ),
    "planning": (
        "Create a concise plan for preparing a small dataset for model fine-tuning.",
        "Plan the steps to migrate a script into a package without changing behavior.",
        "Make a checklist for validating a generated JSONL dataset before training.",
        "Create a rollout plan for enabling a new API endpoint safely.",
        "Plan a one-day debugging session for failing unit tests after a refactor.",
        "Create a checklist for reviewing a pull request that changes data schemas.",
        "Plan the steps to back up and restore a small database.",
        "Create a concise plan for triaging a production latency spike.",
        "Make a checklist for preparing release notes for a minor version update.",
        "Plan how to split a large README into focused documentation files.",
        "Create a practical plan for labeling and archiving old experiment outputs.",
        "Make a checklist for validating cloud cost tags before month-end reporting.",
        "Plan the steps to migrate environment variables into a .env-based workflow.",
        "Create a checklist for testing a new retry policy before production use.",
        "Plan how to compare two teacher models on a small generation bakeoff.",
        "Make a checklist for cleaning a repo before publishing to GitHub.",
        "Create a concise plan for adding structured logging to a Python CLI.",
        "Plan the steps to recover from a failed long-running data generation job.",
        "Make a checklist for inspecting generated preference pairs before training.",
        "Create a practical plan for running a small validation matrix after patches.",
    ),
    "instruction": (
        "Rewrite this sentence to be clearer: The thing was done by the team after the issue happened.",
        "Summarize this in one sentence: The service failed during peak traffic because the database connection pool was exhausted.",
        "Turn this rough note into a clear task: fix bad ids teacher output merge should reject missing duplicate unexpected.",
        "Rewrite this as a direct instruction: maybe look at the logs and see why it broke yesterday.",
        "Turn this note into a concise bug report title: upload path includes scratch files and users see internal batches.",
        "Rewrite this sentence in active voice: The deployment was approved by the review board after testing.",
        "Summarize this incident in one sentence: Requests timed out because the upstream provider rate-limited traffic.",
        "Turn this rough note into a clear TODO item: add provider ignore order controls all generation paths.",
        "Rewrite this instruction to be more specific: make the data better before training.",
        "Convert this note into a checklist item: run tests inspect manifest verify no rejected rows.",
        "Rewrite this sentence to remove ambiguity: They said it was ready after it passed.",
        "Summarize this paragraph in one sentence: The model produced valid JSON but some rows were semantically wrong.",
        "Turn this rough note into a clear commit message: dpo chosen answer wrong validator too weak.",
        "Rewrite this support reply to be clearer: We looked and the issue is probably fixed now.",
        "Make this task concise: check all Make targets because some flags are wired in one path but not others.",
        "Rewrite this sentence for a technical README: Users can do the thing by running the command below.",
        "Summarize this update in one sentence: Routing now avoids a bad upstream provider while keeping fallback enabled.",
        "Turn this note into an acceptance criterion: underfilled run should fail non zero after backfill budget.",
        "Rewrite this warning to be calmer and clearer: don't ever push internal scratch files to hugging face.",
        "Convert this rough plan into one clear instruction: first smoke then validation then small scale then production.",
    ),
}

missing_signals = DISTILLATION_SIGNALS - set(DISTILLATION_PROMPT_SEEDS)
extra_signals = set(DISTILLATION_PROMPT_SEEDS) - DISTILLATION_SIGNALS
if missing_signals or extra_signals:
    raise RuntimeError(
        "distillation prompt seeds must exactly match supported signals; "
        f"missing={sorted(missing_signals)}, extra={sorted(extra_signals)}"
    )


def _normalize_seed_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().strip().split()).strip(" .!?;:")


def _validate_seed_prompt_inventory() -> None:
    problems: list[str] = []
    for signal, prompts in DISTILLATION_PROMPT_SEEDS.items():
        if len(prompts) < MIN_SEED_PROMPTS_PER_SIGNAL:
            problems.append(f"{signal}: only {len(prompts)} prompt(s)")
            continue
        normalized = [_normalize_seed_prompt(prompt) for prompt in prompts]
        if len(set(normalized)) != len(normalized):
            problems.append(f"{signal}: duplicate or near-duplicate seed prompt text")
    if problems:
        raise RuntimeError("distillation prompt seed inventory is not production-diverse enough; " + "; ".join(problems))


_validate_seed_prompt_inventory()


def iter_seed_prompts(signal: str) -> Iterator[str]:
    """Yield built-in seed prompts for one supported distillation signal."""
    normalized_signal = validate_signal(signal)
    yield from DISTILLATION_PROMPT_SEEDS[normalized_signal]


def build_seed_prompt_records(*, signal: str, count: int, start_index: int = 1) -> list[dict[str, object]]:
    """Build deterministic local prompt records by cycling signal seed prompts."""
    normalized_signal = validate_signal(signal)
    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    seeds = DISTILLATION_PROMPT_SEEDS[normalized_signal]
    cycled_prompts = (seeds[index % len(seeds)] for index in range(count))

    records: list[dict[str, object]] = []
    for offset, prompt in enumerate(islice(cycled_prompts, count)):
        records.append(
            build_prompt_record(
                signal=normalized_signal,
                prompt=prompt,
                index=start_index + offset,
                metadata={
                    "seed_source": "builtin",
                    "seed_index": offset % len(seeds),
                },
            )
        )
    return records
