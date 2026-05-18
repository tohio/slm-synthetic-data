
from __future__ import annotations

import hashlib
from typing import Dict, List


def _choose(options: List[str], batch_id: int, salt: str, offset: int = 0) -> str:
    if not options:
        return ""
    digest = hashlib.sha256(f"{salt}:{batch_id}:{offset}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(options)
    return options[idx]


ARITHMETIC_FORMATS = [
    "symbolic equation, e.g. 47 + 29 = ?",
    "short word problem with named objects",
    "missing-value equation, e.g. 12 + ? = 31",
    "comparison question, e.g. which is larger after calculation",
    "two-step arithmetic word problem",
    "unit-like quantity problem without real-world facts",
    "table-free numeric reasoning question",
    "reverse operation check",
]
ARITHMETIC_OPERATIONS = [
    "addition",
    "subtraction",
    "multiplication",
    "exact integer division",
    "mixed addition and subtraction",
    "mixed multiplication and addition",
    "mixed division and subtraction",
]
ARITHMETIC_RANGES = [
    "single-digit numbers",
    "two-digit numbers",
    "three-digit numbers",
    "numbers between 10 and 99",
    "numbers between 100 and 999",
    "small negatives with one operation",
    "multiples of 5, 10, or 25",
    "exact division pairs such as 144 / 12",
]
ARITHMETIC_STYLES = [
    "very concise",
    "plain classroom style",
    "practical shopping/counting style",
    "measurement-style quantities",
    "number puzzle style",
    "check-the-result style",
]

TASK_CODE_TOPICS = [
    "strings and text cleanup",
    "lists and simple aggregation",
    "dictionaries and counting",
    "sets and uniqueness",
    "sorting records",
    "filtering values",
    "loops and conditionals",
    "basic math helper functions",
    "date-like string parsing without external libraries",
    "CSV-like line parsing without file IO",
    "input validation helpers",
    "nested lists",
    "grouping items by key",
    "frequency tables",
    "simple class or dataclass-free object representation",
    "error handling with ValueError",
]
TASK_CODE_PATTERNS = [
    "write one pure function",
    "write a function plus two example calls",
    "write a helper function and a wrapper function",
    "transform a list into a new list",
    "return a dictionary summary",
    "validate inputs before computing",
    "parse a small string format",
    "compute a score or ranking",
]
TASK_CODE_CONSTRAINTS = [
    "avoid imports",
    "use only the Python standard library",
    "keep code under 12 lines",
    "include one edge-case check",
    "do not print except in a tiny example call",
    "use clear variable names",
    "return data instead of printing it",
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
    nonce = hashlib.sha256(f"synthetic-diversity:{signal}:{batch_id}".encode("utf-8")).hexdigest()[:12]

    if signal == "arithmetic":
        return "\n".join([
            f"Batch diversity id: {nonce}",
            f"Primary format: {_choose(ARITHMETIC_FORMATS, batch_id, signal, 1)}",
            f"Operation focus: {_choose(ARITHMETIC_OPERATIONS, batch_id, signal, 2)}",
            f"Number range: {_choose(ARITHMETIC_RANGES, batch_id, signal, 3)}",
            f"Writing style: {_choose(ARITHMETIC_STYLES, batch_id, signal, 4)}",
            "Within this batch, use different numbers, wording, and answer values for every item.",
            "Do not use 2 + 2, 2 + 5, rectangle area, or other common toy examples unless the context specifically requires it.",
        ])

    if signal == "task_code":
        return "\n".join([
            f"Batch diversity id: {nonce}",
            f"Topic focus: {_choose(TASK_CODE_TOPICS, batch_id, signal, 1)}",
            f"Implementation pattern: {_choose(TASK_CODE_PATTERNS, batch_id, signal, 2)}",
            f"Constraint: {_choose(TASK_CODE_CONSTRAINTS, batch_id, signal, 3)}",
            "Within this batch, tasks must solve different problems and use different function names.",
            "Avoid rectangle area, palindrome, factorial, Fibonacci, prime check, and other overused beginner examples unless the topic focus requires them.",
        ])

    if signal == "educational_qa_mcq":
        banned = "; ".join(MCQ_BANNED_EXAMPLES)
        return "\n".join([
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
        ])

    if signal == "factual_restraint":
        return "\n".join([
            f"Batch diversity id: {nonce}",
            f"Uncertainty category: {_choose(FACTUAL_CATEGORIES, batch_id, signal, 1)}",
            f"Safe-answer style: {_choose(FACTUAL_ANSWER_STYLES, batch_id, signal, 2)}",
            "Within this batch, vary the topic, wording, and safe-answer phrasing.",
            "Avoid repeating generic answers such as 'It depends on various factors' unless followed by a specific explanation of what is missing.",
        ])

    return f"Batch diversity id: {nonce}\nUse varied wording, topics, values, and answer formats within this batch."
