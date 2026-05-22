from __future__ import annotations

import hashlib
from typing import Dict, List


def _choose(options: List[str], batch_id: int, salt: str, offset: int = 0) -> str:
    if not options:
        return ""
    digest = hashlib.sha256(f"{salt}:{batch_id}:{offset}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(options)
    return options[idx]


def _choose_profile(
    profiles: List[Dict[str, str]],
    batch_id: int,
    salt: str,
    offset: int = 0,
) -> Dict[str, str]:
    if not profiles:
        return {}
    digest = hashlib.sha256(f"{salt}:{batch_id}:{offset}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(profiles)
    return profiles[idx]


# ---------------------------------------------------------------------------
# Arithmetic diversity profiles
#
# These profiles intentionally preserve direct-equation arithmetic as part of
# the signal, while avoiding concrete operands or reusable example questions.
# Each profile is internally coherent: the format, operation, number design,
# and context belong together.
# ---------------------------------------------------------------------------

ARITHMETIC_PROFILES: List[Dict[str, str]] = [
    {
        "format": "direct symbolic equation",
        "operation": "addition",
        "numbers": "three- or four-digit integers with uncommon operands",
        "context": "none; do not add a story context",
        "answer_focus": "compute the exact sum",
    },
    {
        "format": "direct symbolic equation",
        "operation": "subtraction",
        "numbers": "three- or four-digit integers with uncommon operands",
        "context": "none; do not add a story context",
        "answer_focus": "compute the exact difference",
    },
    {
        "format": "direct symbolic equation",
        "operation": "multiplication",
        "numbers": "a two-digit integer multiplied by a different two- or three-digit integer",
        "context": "none; do not add a story context",
        "answer_focus": "compute the exact product",
    },
    {
        "format": "direct symbolic equation",
        "operation": "exact integer division",
        "numbers": "a non-trivial divisible total and divisor with no remainder",
        "context": "none; do not add a story context",
        "answer_focus": "compute the exact quotient",
    },
    {
        "format": "missing-value equation",
        "operation": "addition relationship",
        "numbers": "a medium-sized total and one known addend",
        "context": "none; use a clear missing-number equation",
        "answer_focus": "return the missing integer",
    },
    {
        "format": "missing-value equation",
        "operation": "subtraction relationship",
        "numbers": "a medium-sized starting value and known result",
        "context": "none; use a clear missing-number equation",
        "answer_focus": "return the missing integer",
    },
    {
        "format": "missing-value scenario",
        "operation": "multiplication relationship",
        "numbers": "a total made from equal groups with one group quantity unknown",
        "context": "package quantities",
        "answer_focus": "return the missing group count or group size",
    },
    {
        "format": "missing-value scenario",
        "operation": "exact division relationship",
        "numbers": "an evenly distributed total with one quantity unknown",
        "context": "team assignments",
        "answer_focus": "return the missing integer",
    },
    {
        "format": "short contextual word problem",
        "operation": "addition",
        "numbers": "two different medium-sized quantities",
        "context": "event ticket allocation",
        "answer_focus": "compute the combined total",
    },
    {
        "format": "short contextual word problem",
        "operation": "subtraction",
        "numbers": "a medium-sized inventory total and a distinct removed quantity",
        "context": "warehouse item movement",
        "answer_focus": "compute the remaining amount",
    },
    {
        "format": "short contextual word problem",
        "operation": "multiplication",
        "numbers": "multiple equal groups using non-trivial group counts",
        "context": "production batches",
        "answer_focus": "compute the total produced quantity",
    },
    {
        "format": "short contextual word problem",
        "operation": "exact integer division",
        "numbers": "an evenly divisible total and group count",
        "context": "meal-box distribution",
        "answer_focus": "compute the equal amount per group",
    },
    {
        "format": "two-step contextual scenario",
        "operation": "multiplication followed by addition",
        "numbers": "equal groups plus a separate extra quantity",
        "context": "package shipment quantities",
        "answer_focus": "compute the final total",
    },
    {
        "format": "two-step contextual scenario",
        "operation": "multiplication followed by subtraction",
        "numbers": "equal groups followed by a removed quantity",
        "context": "parking-space usage",
        "answer_focus": "compute the remaining total",
    },
    {
        "format": "two-step contextual scenario",
        "operation": "addition followed by subtraction",
        "numbers": "two incoming quantities followed by an outgoing quantity",
        "context": "library checkout counts",
        "answer_focus": "compute the final count",
    },
    {
        "format": "two-step contextual scenario",
        "operation": "exact division followed by addition",
        "numbers": "an evenly divided amount plus an additional amount",
        "context": "resource planning",
        "answer_focus": "compute the final amount",
    },
    {
        "format": "comparison question",
        "operation": "compare two addition totals",
        "numbers": "different medium-sized operand pairs",
        "context": "delivery-route quantities",
        "answer_focus": "return the larger computed integer",
    },
    {
        "format": "comparison question",
        "operation": "compare a multiplication total with an addition total",
        "numbers": "easy-to-verify but non-trivial integers",
        "context": "classroom material counts",
        "answer_focus": "return the larger computed integer",
    },
    {
        "format": "comparison question",
        "operation": "compare two remaining quantities after subtraction",
        "numbers": "different starting and removed values",
        "context": "store inventory changes",
        "answer_focus": "return the smaller or larger computed integer as requested",
    },
    {
        "format": "ordering question",
        "operation": "order three computed totals from addition and subtraction",
        "numbers": "three distinct calculations using medium-sized integers",
        "context": "game-score changes",
        "answer_focus": "return the requested extreme value or ordered numeric results",
    },
    {
        "format": "ordering question",
        "operation": "order three grouped totals",
        "numbers": "three distinct multiplication calculations",
        "context": "crate and carton quantities",
        "answer_focus": "return the requested extreme value or ordered numeric results",
    },
    {
        "format": "allocation scenario",
        "operation": "exact integer division",
        "numbers": "a larger total divided evenly among several groups",
        "context": "bus or train seat allocation",
        "answer_focus": "compute the per-group allocation",
    },
    {
        "format": "budget scenario",
        "operation": "multiplication followed by subtraction",
        "numbers": "a total cost calculation followed by a budget difference",
        "context": "simple purchasing plan",
        "answer_focus": "compute the remaining amount or shortfall",
    },
    {
        "format": "schedule scenario",
        "operation": "multiplication followed by addition",
        "numbers": "repeated time blocks plus one additional block",
        "context": "time-block allocation",
        "answer_focus": "compute the final number of units",
    },
]

ARITHMETIC_STYLES = [
    "compact calculation steps",
    "plain classroom wording",
    "practical quantity wording",
    "short verification wording",
    "concise planning wording",
    "measurement-like quantity wording",
    "clear compare-or-order wording",
]


# ---------------------------------------------------------------------------
# Task-code diversity profiles
#
# These profiles align with the function-only task_code prompt:
# - exactly one complete function
# - no imports, print calls, examples, f-strings, exceptions, file I/O,
#   regex, CSV parsing, date parsing, or helper/wrapper pairs
# - no reusable concrete function names or code examples in the prompt
# ---------------------------------------------------------------------------

TASK_CODE_PROFILES: List[Dict[str, str]] = [
    {
        "family": "build a frequency summary",
        "input_shape": "list of short string labels",
        "output_shape": "dictionary mapping labels to counts",
        "implementation": "one loop with a dictionary accumulator",
        "domain": "message tags",
    },
    {
        "family": "build a frequency summary",
        "input_shape": "list of category strings",
        "output_shape": "dictionary mapping categories to counts",
        "implementation": "one loop with conditional dictionary updates",
        "domain": "package statuses",
    },
    {
        "family": "sum values by group",
        "input_shape": "list of dictionaries containing a category and numeric amount",
        "output_shape": "dictionary mapping each category to its total amount",
        "implementation": "one loop with a dictionary accumulator",
        "domain": "order quantities",
    },
    {
        "family": "sum values by group",
        "input_shape": "list of dictionaries containing a label and score",
        "output_shape": "dictionary mapping each label to its total score",
        "implementation": "one loop with dictionary updates",
        "domain": "team totals",
    },
    {
        "family": "filter structured records by a condition",
        "input_shape": "list of dictionaries with a numeric score field",
        "output_shape": "list of records meeting a numeric threshold",
        "implementation": "one list comprehension with a comparison",
        "domain": "student results",
    },
    {
        "family": "filter structured records by a condition",
        "input_shape": "list of dictionaries with a status field",
        "output_shape": "list of records with the selected status",
        "implementation": "one loop that appends matching records",
        "domain": "delivery items",
    },
    {
        "family": "filter numeric values",
        "input_shape": "list of integers",
        "output_shape": "list of values meeting a numeric condition",
        "implementation": "one list comprehension with arithmetic and comparison",
        "domain": "daily measurements",
    },
    {
        "family": "transform numeric values",
        "input_shape": "list of integers",
        "output_shape": "list of adjusted integers",
        "implementation": "one loop applying a simple arithmetic transformation",
        "domain": "score adjustments",
    },
    {
        "family": "transform structured records",
        "input_shape": "list of dictionaries with a numeric quantity field",
        "output_shape": "list of computed numeric values",
        "implementation": "one list comprehension using one field from each record",
        "domain": "cart quantities",
    },
    {
        "family": "sort structured records",
        "input_shape": "list of dictionaries with one numeric ordering field",
        "output_shape": "sorted list of the original records",
        "implementation": "one call to sorted with a simple key function",
        "domain": "task priorities",
    },
    {
        "family": "sort structured records",
        "input_shape": "list of dictionaries with one short string ordering field",
        "output_shape": "sorted list of the original records",
        "implementation": "one call to sorted with a simple key function",
        "domain": "label records",
    },
    {
        "family": "sort simple values after a transformation",
        "input_shape": "list of short strings",
        "output_shape": "list sorted according to a computed property",
        "implementation": "one call to sorted with a simple key function",
        "domain": "short messages",
    },
    {
        "family": "collect distinct values from structured records",
        "input_shape": "list of dictionaries with a category field",
        "output_shape": "sorted list of distinct category values",
        "implementation": "set accumulation followed by sorted output",
        "domain": "inventory records",
    },
    {
        "family": "measure distinct values",
        "input_shape": "list of short string values",
        "output_shape": "integer count of distinct values",
        "implementation": "set accumulation with an additional simple condition",
        "domain": "badge labels",
    },
    {
        "family": "group structured records by one field",
        "input_shape": "list of dictionaries containing a group field",
        "output_shape": "dictionary mapping group values to lists of records",
        "implementation": "one loop using dictionary list accumulation",
        "domain": "shipment records",
    },
    {
        "family": "group simple strings by a derived property",
        "input_shape": "list of non-empty short strings",
        "output_shape": "dictionary mapping a derived key to lists of strings",
        "implementation": "one loop using a simple string property",
        "domain": "category labels",
    },
    {
        "family": "compute an aggregate metric",
        "input_shape": "list of integers",
        "output_shape": "integer total after applying a condition",
        "implementation": "one loop with a numeric accumulator and conditional",
        "domain": "activity counts",
    },
    {
        "family": "compute an aggregate metric",
        "input_shape": "list of dictionaries with a numeric field",
        "output_shape": "integer count of records meeting a threshold",
        "implementation": "one loop with a counter",
        "domain": "registration records",
    },
    {
        "family": "find an extreme structured value",
        "input_shape": "list of dictionaries with a numeric field",
        "output_shape": "one selected record",
        "implementation": "one loop tracking the current best record",
        "domain": "product metrics",
    },
    {
        "family": "find an extreme numeric value with filtering",
        "input_shape": "list of integers",
        "output_shape": "one selected integer",
        "implementation": "filter eligible values in a loop while tracking the best",
        "domain": "performance scores",
    },
    {
        "family": "combine two dictionaries",
        "input_shape": "two dictionaries mapping labels to integer counts",
        "output_shape": "new dictionary containing combined totals",
        "implementation": "one loop over each input dictionary with numeric accumulation",
        "domain": "warehouse totals",
    },
    {
        "family": "compare paired values",
        "input_shape": "two equally sized lists of integers",
        "output_shape": "list containing selected values from each pair",
        "implementation": "one loop over paired values using a comparison",
        "domain": "weekly measurements",
    },
    {
        "family": "flatten nested values with filtering",
        "input_shape": "nested list of integers",
        "output_shape": "flat list containing values meeting a simple condition",
        "implementation": "nested loops with conditional append",
        "domain": "batch readings",
    },
    {
        "family": "summarize a one-line string",
        "input_shape": "single one-line text value containing space-separated words",
        "output_shape": "dictionary mapping normalized words to counts",
        "implementation": "split text then count tokens in one loop",
        "domain": "short notes",
    },
]

TASK_CODE_CONSTRAINTS = [
    "Use exactly one complete function definition and no top-level calls.",
    "Do not use imports, printing, examples, exceptions, formatted strings, or external packages.",
    "Keep the implementation compact but include meaningful logic beyond a trivial built-in wrapper.",
    "Use clear variable names that do not copy familiar beginner-example function names.",
    "Return computed data without mutating inputs unless the requested output requires a new accumulator.",
    "Do not reuse common one-line solutions or generic task titles.",
]


# ---------------------------------------------------------------------------
# MCQ diversity settings: unchanged.
# ---------------------------------------------------------------------------

MCQ_SUBJECTS = [
    "integer arithmetic",
    "fractions and decimals",
    "percentages and ratios",
    "basic algebra equations",
    "geometry shapes and area",
    "measurement and units",
    "physical science forces and motion",
    "earth science weather and rocks",
    "biology cells and ecosystems",
    "computer science algorithms",
    "Python data types",
    "Python control flow",
    "history timelines and sources",
    "geography maps and regions",
    "grammar and sentence structure",
    "vocabulary in context",
    "logic puzzles and inference",
    "data interpretation from small tables",
    "technology literacy and privacy",
    "reading comprehension",
    "scientific method",
    "basic probability",
    "statistics averages and spread",
    "financial literacy basics",
]
MCQ_LEVELS = [
    "upper elementary",
    "middle school",
    "high school intro",
    "adult beginner",
    "mixed-review beginner",
]
MCQ_STYLES = [
    "definition question with a concrete example",
    "worked example question using non-trivial values",
    "identify the best explanation",
    "classification question with four distinct categories",
    "cause-and-effect question",
    "choose the next step in a process",
    "spot the misconception",
    "short scenario question with named objects",
    "compare two options and choose the better answer",
    "interpret a tiny table or list described in text",
    "fill in the missing step",
    "choose the statement that must be true",
]
MCQ_CONTEXTS = [
    "classroom practice",
    "workplace note",
    "science lab observation",
    "small business example",
    "sports or games example",
    "garden or nature example",
    "library or reading example",
    "travel or map example",
    "simple coding exercise",
    "home budget example",
    "weather report example",
    "recipe or measurement example",
]
MCQ_STEM_PATTERNS = [
    "Start with 'A student...' and ask for the best answer.",
    "Start with 'Which statement...' and ask for the correct statement.",
    "Start with 'In this example...' and include concrete details.",
    "Start with 'What should happen next...' and ask for a next step.",
    "Start with 'Why...' and ask for an explanation, not a bare fact.",
    "Start with 'Which choice best describes...' and compare concepts.",
    "Start with 'Given...' and include a small value, list, or condition.",
    "Start with 'A teacher asks...' and include a specific scenario.",
]
MCQ_DISTRACTOR_STRATEGIES = [
    "use common misconceptions as distractors",
    "use near-miss numeric answers",
    "use same-category but wrong concepts",
    "use plausible but incomplete explanations",
    "use wrong order or wrong next step",
    "use overgeneralized statements as distractors",
]
MCQ_BANNED_EXAMPLES = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "What color is the sky?",
    "Which planet is known as the Red Planet?",
    "What is the largest mammal?",
    "Who wrote Hamlet?",
    "What is photosynthesis?",
]


# ---------------------------------------------------------------------------
# Factual-restraint diversity settings: unchanged.
# ---------------------------------------------------------------------------

FACTUAL_CATEGORIES = [
    "unknown future event",
    "private or unavailable information",
    "fictional premise presented as real",
    "overly broad exhaustive list request",
    "scientific uncertainty",
    "historical claim with insufficient detail",
    "ambiguous person/place/entity",
    "unverifiable statistic",
    "secret recipe or proprietary information",
    "medical/legal/financial uncertainty without context",
    "counterfactual scenario",
    "internet rumor",
]
FACTUAL_ANSWER_STYLES = [
    "brief refusal to speculate",
    "explain what information would be needed",
    "state uncertainty and avoid guessing",
    "offer a safe general framing",
    "ask for clarification without inventing facts",
    "distinguish known facts from unknown details",
]


def build_diversity_context(signal: str, batch_id: int) -> str:
    """Return deterministic per-batch diversity constraints for prompt construction."""
    signal = signal.strip().lower()
    nonce = hashlib.sha256(
        f"synthetic-diversity:{signal}:{batch_id}".encode("utf-8")
    ).hexdigest()[:12]

    if signal == "arithmetic":
        profile = _choose_profile(ARITHMETIC_PROFILES, batch_id, signal, 1)
        style = _choose(ARITHMETIC_STYLES, batch_id, signal, 2)

        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                f"Required format: {profile['format']}",
                f"Required operation structure: {profile['operation']}",
                f"Number design: {profile['numbers']}",
                f"Context guidance: {profile['context']}",
                f"Answer focus: {profile['answer_focus']}",
                f"Writing style: {style}",
                "Treat this as a coherent profile: do not replace it with a different arithmetic form.",
                "Use new operands, wording, entities, and answer values for every record.",
                "Direct equations remain allowed only when this profile explicitly requires them.",
                "Do not copy concrete operands, stories, or phrasing from instructions or familiar textbook examples.",
                "Do not include the batch diversity id in any generated field.",
            ]
        )

    if signal == "task_code":
        profile = _choose_profile(TASK_CODE_PROFILES, batch_id, signal, 1)
        constraint = _choose(TASK_CODE_CONSTRAINTS, batch_id, signal, 2)

        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                f"Required task family: {profile['family']}",
                f"Required input shape: {profile['input_shape']}",
                f"Required output shape: {profile['output_shape']}",
                f"Required implementation approach: {profile['implementation']}",
                f"Domain guidance: {profile['domain']}",
                f"Additional constraint: {constraint}",
                "Treat this as a coherent profile: the task, input, output, and implementation must agree.",
                "Generate exactly one complete Python function definition and no example calls.",
                "Do not import modules, print results, use exceptions, use f-strings, or generate parsing/regex/file-I/O tasks.",
                "Use a distinct task title, function name, variable naming scheme, and implementation body.",
                "Do not copy task names or function bodies from the prompt or from familiar beginner examples.",
                "Do not include the batch diversity id in any generated field.",
            ]
        )

    if signal == "educational_qa_mcq":
        banned = "; ".join(MCQ_BANNED_EXAMPLES)
        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                f"Subject focus: {_choose(MCQ_SUBJECTS, batch_id, signal, 1)}",
                f"Level: {_choose(MCQ_LEVELS, batch_id, signal, 2)}",
                f"Question style: {_choose(MCQ_STYLES, batch_id, signal, 3)}",
                f"Scenario context: {_choose(MCQ_CONTEXTS, batch_id, signal, 4)}",
                f"Stem pattern: {_choose(MCQ_STEM_PATTERNS, batch_id, signal, 5)}",
                f"Distractor strategy: {_choose(MCQ_DISTRACTOR_STRATEGIES, batch_id, signal, 6)}",
                f"Correct-index rotation starts at: {int(hashlib.sha256((signal + str(batch_id)).encode('utf-8')).hexdigest()[:2], 16) % 4}",
                "Within this batch, every question must use a different stem, topic detail, and correct answer.",
                "For numeric questions, avoid tiny toy values; prefer two-step, fractions, percentages, ratios, or non-obvious values.",
                "For non-numeric questions, include a concrete scenario detail instead of asking a generic definition.",
                "Do not reuse answer choices across items in the same batch.",
                "Do not include the batch diversity id in any generated field.",
                f"Banned examples: {banned}",
            ]
        )

    if signal == "factual_restraint":
        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                f"Uncertainty category: {_choose(FACTUAL_CATEGORIES, batch_id, signal, 1)}",
                f"Safe-answer style: {_choose(FACTUAL_ANSWER_STYLES, batch_id, signal, 2)}",
                "Within this batch, vary the topic, wording, and safe-answer phrasing.",
                "Avoid repeating generic answers such as 'It depends on various factors' unless followed by a specific explanation of what is missing.",
            ]
        )

    return (
        f"Batch diversity id: {nonce}\n"
        "Use varied wording, topics, values, and answer formats within this batch."
    )