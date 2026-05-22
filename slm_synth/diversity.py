from __future__ import annotations

import hashlib
from typing import List


def _choose(options: List[str], batch_id: int, salt: str, offset: int = 0) -> str:
    if not options:
        return ""
    digest = hashlib.sha256(f"{salt}:{batch_id}:{offset}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(options)
    return options[idx]


ARITHMETIC_FORMATS = [
    "direct symbolic equation using uncommon operands",
    "short contextual word problem",
    "missing-value equation",
    "comparison of two computed integer values",
    "ordering of three computed integer values",
    "two-step arithmetic scenario",
    "equal-sharing question with exact integer division",
    "reverse-operation check",
    "grouped quantity calculation",
    "budget or allocation calculation",
]
ARITHMETIC_OPERATIONS = [
    "addition",
    "subtraction",
    "multiplication",
    "exact integer division",
    "mixed addition and subtraction",
    "multiplication followed by addition",
    "multiplication followed by subtraction",
    "division followed by addition",
    "comparison using two different operations",
    "ordering using multiple operations",
]
ARITHMETIC_RANGES = [
    "small integers with varied operands",
    "two-digit integers avoiding repeated textbook pairs",
    "three-digit integers",
    "four-digit totals with easy verification",
    "numbers between 100 and 999",
    "numbers between 1,000 and 9,999",
    "clear negative result from subtraction or comparison",
    "exact integer division using non-trivial factor pairs",
    "mixed ranges with a larger total and smaller adjustment",
]
ARITHMETIC_CONTEXTS = [
    "package shipment quantities",
    "event ticket allocation",
    "classroom material counts",
    "warehouse item totals",
    "bus or train seat allocation",
    "meal-box distribution",
    "delivery-route quantities",
    "parking-space usage",
    "store receipt totals",
    "production batch counts",
    "library checkout counts",
    "game-score changes",
    "time-block allocation",
    "team assignment counts",
    "forms, labels, badges, or cards",
    "crates, cartons, bins, or pallets",
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

TASK_CODE_TOPICS = [
    "aggregate values from a list",
    "count labels in a dictionary summary",
    "measure uniqueness in a collection",
    "sort simple values",
    "sort structured records by one field",
    "filter records by one condition",
    "transform list values with a loop",
    "apply a numeric calculation to a sequence",
    "split simple one-line text without punctuation parsing",
    "group records by a key",
    "create a frequency summary",
    "flatten or inspect nested lists",
    "combine two collections with simple logic",
    "select values meeting a numeric threshold",
    "compute a small metric from structured data",
]
TASK_CODE_PATTERNS = [
    "define exactly one pure function returning a list",
    "define exactly one pure function returning a dictionary",
    "define exactly one pure function returning an integer",
    "define exactly one pure function returning a boolean",
    "define exactly one function using a loop and accumulator",
    "define exactly one function using a dictionary update pattern",
    "define exactly one function using a list comprehension",
    "define exactly one function using sorted with a simple key",
    "define exactly one function using a set as an intermediate value",
    "define exactly one function using conditional filtering",
]
TASK_CODE_DATA_SHAPES = [
    "list of integers",
    "list of short strings",
    "list of dictionaries with one simple field",
    "dictionary of labels to integer counts",
    "dictionary of names to numeric values",
    "nested list of small values",
    "single one-line string",
    "pair of simple lists",
    "list of tuples represented as lists",
]
TASK_CODE_DOMAINS = [
    "inventory labels",
    "student scores",
    "event registrations",
    "package statuses",
    "product categories",
    "task priorities",
    "seat assignments",
    "message tags",
    "daily counts",
    "order quantities",
    "route stops",
    "badge records",
    "book categories",
    "team totals",
    "form statuses",
]
TASK_CODE_CONSTRAINTS = [
    "use exactly one function definition and no top-level calls",
    "avoid imports, printing, examples, exceptions, and formatted strings",
    "keep the implementation under 12 lines",
    "return computed data instead of mutating external state",
    "use clear variable names without copying generic examples",
    "include meaningful logic beyond wrapping a single built-in call",
    "avoid repeated function names and repeated one-line solutions",
]

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
        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                f"Primary format: {_choose(ARITHMETIC_FORMATS, batch_id, signal, 1)}",
                f"Operation focus: {_choose(ARITHMETIC_OPERATIONS, batch_id, signal, 2)}",
                f"Number range: {_choose(ARITHMETIC_RANGES, batch_id, signal, 3)}",
                f"Context domain: {_choose(ARITHMETIC_CONTEXTS, batch_id, signal, 4)}",
                f"Writing style: {_choose(ARITHMETIC_STYLES, batch_id, signal, 5)}",
                "Use new operands, entities, wording, operation structure, and answer values for this record.",
                "Direct equations are allowed when selected, but use uncommon operands and do not copy any example from instructions.",
                "For contextual problems, avoid generic textbook stories and vary the nouns and scenario structure.",
                "Do not include the batch diversity id in any generated field.",
            ]
        )

    if signal == "task_code":
        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                f"Task family: {_choose(TASK_CODE_TOPICS, batch_id, signal, 1)}",
                f"Implementation shape: {_choose(TASK_CODE_PATTERNS, batch_id, signal, 2)}",
                f"Input data shape: {_choose(TASK_CODE_DATA_SHAPES, batch_id, signal, 3)}",
                f"Domain context: {_choose(TASK_CODE_DOMAINS, batch_id, signal, 4)}",
                f"Constraint: {_choose(TASK_CODE_CONSTRAINTS, batch_id, signal, 5)}",
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