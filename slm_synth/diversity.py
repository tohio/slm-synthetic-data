from __future__ import annotations

import hashlib
from typing import Dict, List, Union


ProfileValue = Union[str, List[str]]


def _choose(options: List[str], batch_id: int, salt: str, offset: int = 0) -> str:
    if not options:
        return ""
    digest = hashlib.sha256(f"{salt}:{batch_id}:{offset}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(options)
    return options[idx]


def _choose_profile(
    profiles: List[Dict[str, ProfileValue]],
    batch_id: int,
    salt: str,
    offset: int = 0,
) -> Dict[str, ProfileValue]:
    if not profiles:
        return {}
    digest = hashlib.sha256(f"{salt}:{batch_id}:{offset}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(profiles)
    return profiles[idx]


def _choose_int(batch_id: int, salt: str, offset: int, minimum: int, maximum: int) -> int:
    digest = hashlib.sha256(f"{salt}:{batch_id}:{offset}".encode("utf-8")).hexdigest()
    return minimum + (int(digest[:8], 16) % (maximum - minimum + 1))


# ---------------------------------------------------------------------------
# Arithmetic diversity profiles
#
# Direct-equation arithmetic remains part of the signal. These profiles avoid
# fixed example operands and keep each selected format internally coherent.
# ---------------------------------------------------------------------------

ARITHMETIC_PROFILES: List[Dict[str, ProfileValue]] = [
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
# Every profile requires one complete Python function only. Profiles require
# an additional transformation, filter, comparison, or aggregation step to
# avoid collapse into trivial canonical snippets at scale.
# ---------------------------------------------------------------------------

TASK_CODE_PROFILES: List[Dict[str, ProfileValue]] = [
    {
        "family": "conditionally total values by group",
        "input_shape": "list of dictionaries containing a group label, numeric amount, and simple status field",
        "output_shape": "dictionary mapping qualifying groups to summed amounts",
        "implementation": "one loop that filters by status before updating group totals",
        "domains": ["shipment entries", "equipment requests", "resource allocations", "supply movements"],
        "rule": "Only include records whose status matches the selected one-word category.",
    },
    {
        "family": "conditionally count records by category",
        "input_shape": "list of dictionaries containing a category and numeric quantity",
        "output_shape": "dictionary mapping categories to counts of qualifying records",
        "implementation": "one loop that tests a numeric cutoff before incrementing category counts",
        "domains": ["service requests", "material batches", "attendance entries", "reservation records"],
        "rule": "Use the assigned cutoff and count records whose quantity meets or exceeds it.",
    },
    {
        "family": "aggregate adjusted values by label",
        "input_shape": "list of dictionaries containing a label and numeric value",
        "output_shape": "dictionary mapping labels to totals after applying an adjustment",
        "implementation": "one loop that adjusts each numeric value then accumulates it by label",
        "domains": ["usage entries", "allocation records", "survey tallies", "workload records"],
        "rule": "Apply the assigned integer adjustment before accumulating values.",
    },
    {
        "family": "filter then sort structured records",
        "input_shape": "list of dictionaries with a numeric measure and a short text identifier",
        "output_shape": "list of qualifying records sorted by the numeric measure",
        "implementation": "one list comprehension for filtering followed by sorted with a simple key",
        "domains": ["quality checks", "capacity readings", "duration records", "inventory observations"],
        "rule": "Filter using the assigned cutoff and sort in the assigned direction.",
    },
    {
        "family": "sort records after deriving a numeric measure",
        "input_shape": "list of dictionaries containing a short text field and another identifier",
        "output_shape": "sorted list of the original records",
        "implementation": "one call to sorted using the length of the selected text field as the key",
        "domains": ["request labels", "route descriptions", "catalog titles", "annotation records"],
        "rule": "Use text length as the ordering measure and sort in the assigned direction.",
    },
    {
        "family": "transform then filter numeric values",
        "input_shape": "list of integers",
        "output_shape": "list of transformed integers that satisfy a condition",
        "implementation": "one loop that transforms each value and appends only qualifying results",
        "domains": ["sensor readings", "production measurements", "distance samples", "daily totals"],
        "rule": "Apply the assigned multiplier and offset, then keep transformed values meeting the cutoff.",
    },
    {
        "family": "calculate bucketed frequencies from numbers",
        "input_shape": "list of integers",
        "output_shape": "dictionary mapping bucket labels to counts",
        "implementation": "one loop that assigns each number to one of two buckets using a cutoff",
        "domains": ["load estimates", "session durations", "queue sizes", "volume readings"],
        "rule": "Use the assigned cutoff to produce two clearly named buckets.",
    },
    {
        "family": "group strings by a derived key after normalization",
        "input_shape": "list of non-empty short strings",
        "output_shape": "dictionary mapping a derived key to lists of normalized strings",
        "implementation": "one loop that normalizes each string and groups it by a simple derived property",
        "domains": ["topic labels", "reference codes", "short descriptors", "region tags"],
        "rule": "Normalize case and group by the assigned derived-key rule.",
    },
    {
        "family": "count normalized tokens meeting a length rule",
        "input_shape": "single one-line text string",
        "output_shape": "dictionary mapping qualifying normalized tokens to counts",
        "implementation": "split into tokens, normalize each token, and count only tokens satisfying a length cutoff",
        "domains": ["short summaries", "status notes", "category descriptions", "single-line comments"],
        "rule": "Use the assigned minimum token length and ignore shorter tokens.",
    },
    {
        "family": "find the best qualifying structured record",
        "input_shape": "list of dictionaries containing a numeric measure and an identifier",
        "output_shape": "one selected record or an empty result when none qualify",
        "implementation": "one loop that applies a cutoff and tracks the best qualifying record",
        "domains": ["inspection results", "request estimates", "route metrics", "resource options"],
        "rule": "Use the assigned cutoff and direction when selecting the best record.",
    },
    {
        "family": "combine two numeric summaries with a rule",
        "input_shape": "two dictionaries mapping labels to integer quantities",
        "output_shape": "new dictionary containing combined values that meet a condition",
        "implementation": "accumulate both dictionaries into a new dictionary, then filter by a cutoff",
        "domains": ["monthly totals", "location counts", "category tallies", "stock movements"],
        "rule": "Use the assigned cutoff to decide which combined totals remain in the output.",
    },
    {
        "family": "compare paired numeric sequences",
        "input_shape": "two equal-length lists of integers",
        "output_shape": "dictionary summarizing pairwise comparison outcomes",
        "implementation": "one loop over paired values that increments outcome counters",
        "domains": ["trial readings", "weekly counts", "capacity snapshots", "forecast comparisons"],
        "rule": "Count greater-than, less-than, and equal outcomes without returning raw pairs.",
    },
    {
        "family": "flatten nested values while applying a numeric rule",
        "input_shape": "nested list of integers",
        "output_shape": "flat list of adjusted values satisfying a condition",
        "implementation": "nested loops that transform and conditionally append values",
        "domains": ["batch outputs", "group measurements", "daily segments", "partition totals"],
        "rule": "Apply the assigned offset and retain values meeting the assigned divisibility rule.",
    },
    {
        "family": "summarize records using a derived group and numeric total",
        "input_shape": "list of dictionaries containing a short label and numeric amount",
        "output_shape": "dictionary mapping derived groups to summed amounts",
        "implementation": "one loop that derives a group from the label then accumulates amounts",
        "domains": ["item codes", "region labels", "request tags", "asset categories"],
        "rule": "Derive groups using the assigned text-position rule.",
    },
    {
        "family": "select labels by combined conditions",
        "input_shape": "dictionary mapping short labels to integer values",
        "output_shape": "sorted list of labels whose values satisfy two simple conditions",
        "implementation": "one list comprehension followed by sorted output",
        "domains": ["category totals", "daily usage", "allocation limits", "record counts"],
        "rule": "Apply the assigned cutoff and parity requirement before sorting labels.",
    },
    {
        "family": "calculate changes between aligned numeric sequences",
        "input_shape": "two equal-length lists of integers",
        "output_shape": "list of signed differences that meet a magnitude condition",
        "implementation": "one loop that computes differences and filters by absolute magnitude",
        "domains": ["before-after readings", "weekly adjustments", "allocation changes", "usage changes"],
        "rule": "Use the assigned minimum magnitude for retained changes.",
    },
]

TASK_CODE_CONSTRAINTS = [
    "Use exactly one complete function definition and no top-level calls.",
    "Do not use imports, printing, examples, exceptions, formatted strings, or external packages.",
    "Use the assigned rule in the implementation; do not simplify it into a generic one-line task.",
    "Keep the implementation compact while preserving the required transformation and condition.",
    "Use distinct field names and function naming; avoid familiar beginner-example names.",
    "Return new computed data without mutating input objects.",
]

TASK_CODE_GROUP_RULES = [
    "the first normalized character",
    "the last normalized character",
    "the first two normalized characters",
    "a short-versus-long label bucket using the assigned cutoff",
]

TASK_CODE_SORT_DIRECTIONS = ["ascending order", "descending order"]
TASK_CODE_STATUSES = ["selected", "ready", "reviewed", "complete", "approved"]


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
        domains = profile.get("domains", [])
        domain = _choose(domains if isinstance(domains, list) else [], batch_id, signal, 3)
        cutoff = _choose_int(batch_id, signal, 4, 11, 97)
        secondary = _choose_int(batch_id, signal, 5, 2, 13)
        divisor = _choose_int(batch_id, signal, 6, 2, 9)
        direction = _choose(TASK_CODE_SORT_DIRECTIONS, batch_id, signal, 7)
        status = _choose(TASK_CODE_STATUSES, batch_id, signal, 8)
        group_rule = _choose(TASK_CODE_GROUP_RULES, batch_id, signal, 9)

        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                f"Required task family: {profile['family']}",
                f"Required input shape: {profile['input_shape']}",
                f"Required output shape: {profile['output_shape']}",
                f"Required implementation approach: {profile['implementation']}",
                f"Domain guidance: {domain}",
                f"Required rule: {profile['rule']}",
                (
                    "Record-specific variation requirements: "
                    f"primary cutoff={cutoff}; secondary integer={secondary}; "
                    f"divisor={divisor}; ordering={direction}; status label={status}; "
                    f"derived-key rule={group_rule}."
                ),
                f"Additional constraint: {constraint}",
                "Use only the record-specific parameters relevant to the selected task family.",
                "Treat this as a coherent profile: the task, input, output, rule, and implementation must agree.",
                "Generate exactly one complete Python function definition and no example calls.",
                "Do not import modules, print results, use exceptions, use f-strings, or generate parsing/regex/file-I/O tasks.",
                "Use a distinct task title, function name, variable naming scheme, and implementation body.",
                "Do not fall back to simple temperature filters, package-status frequencies, task-priority sorts, score scaling, or other common repeated beginner tasks.",
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
