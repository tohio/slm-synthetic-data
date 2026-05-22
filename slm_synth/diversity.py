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
# MCQ verified numeric profiles
#
# This signal is temporarily restricted to numeric question families whose
# indexed answer can be machine-checked before publication.
# ---------------------------------------------------------------------------

MCQ_MATH_VERIFIED_PROFILES: List[Dict[str, ProfileValue]] = [
    {
        "family": "integer addition or subtraction",
        "question_shape": "ask for the exact result of one operation using two non-trivial integers",
        "expression_shape": "one addition or subtraction expression",
        "context": "plain numeric reasoning",
    },
    {
        "family": "integer multiplication",
        "question_shape": "ask for the exact product of two non-trivial integer quantities",
        "expression_shape": "one multiplication expression",
        "context": "grouped supply counts",
    },
    {
        "family": "exact integer division",
        "question_shape": "ask for an equal share from an exactly divisible integer total",
        "expression_shape": "one division expression whose result is an integer",
        "context": "allocation groups",
    },
    {
        "family": "two-step integer arithmetic",
        "question_shape": "provide two incoming quantities and one outgoing quantity, then ask for the final count",
        "expression_shape": "addition followed by subtraction",
        "context": "inventory movement",
    },
    {
        "family": "rectangle area",
        "question_shape": "provide both integer length and integer width, then ask for area",
        "expression_shape": "length multiplied by width",
        "context": "measurement practice",
    },
    {
        "family": "rectangle perimeter",
        "question_shape": "provide both integer length and integer width, then ask for perimeter",
        "expression_shape": "two multiplied by the sum of length and width",
        "context": "measurement practice",
    },
    {
        "family": "fraction of a whole quantity",
        "question_shape": "give an integer total and a fractional part selected so the result is an integer",
        "expression_shape": "total multiplied by numerator then divided by denominator",
        "context": "items in a collection",
    },
    {
        "family": "percentage of a whole quantity",
        "question_shape": "give an integer total and a percentage selected so the result is an integer",
        "expression_shape": "total multiplied by percent then divided by one hundred",
        "context": "completed units",
    },
    {
        "family": "ratio share",
        "question_shape": "give a two-part ratio and a total divisible by the sum of ratio parts, then ask for one share",
        "expression_shape": "total multiplied by requested part then divided by sum of parts",
        "context": "distributed materials",
    },
    {
        "family": "tiny table total or difference",
        "question_shape": "describe three integer values and ask for a total or a difference between computed totals",
        "expression_shape": "a short addition or addition-and-subtraction expression",
        "context": "daily counts",
    },
    {
        "family": "integer average",
        "question_shape": "give three integer values whose arithmetic mean is an integer, then ask for the mean",
        "expression_shape": "sum of values divided by three",
        "context": "measurement summary",
    },
]

MCQ_MATH_DIFFICULTIES = [
    "upper elementary with non-trivial integers",
    "middle school review with two-step reasoning",
    "adult beginner practical numeracy",
]

MCQ_MATH_DISTRACTOR_RULES = [
    "Use three nearby integer distractors from common arithmetic mistakes.",
    "Use three distinct integer distractors based on omitted or reversed operations.",
    "Use three plausible but incorrect integer results; none may equal the verified answer.",
]

# ---------------------------------------------------------------------------
# General non-math MCQ diversity profiles
#
# General MCQs deliberately exclude numeric computation. Each profile requires
# the supporting evidence to appear inside the question itself.
# ---------------------------------------------------------------------------

MCQ_GENERAL_PROFILES: List[Dict[str, ProfileValue]] = [
    {
        "subject": "Python collection behavior",
        "construction": "include a short Python expression or snippet and ask what non-numeric property it demonstrates",
        "evidence": "the code fragment in the question",
        "context": "small programming example",
    },
    {
        "subject": "Python control-flow interpretation",
        "construction": "include a short conditional or loop condition and ask which branch or stopping rule it represents",
        "evidence": "the supplied code condition",
        "context": "simple coding exercise",
    },
    {
        "subject": "computer-science concept",
        "construction": "describe a small operation on data and ask which concept it illustrates",
        "evidence": "the operation described in the question",
        "context": "data organization scenario",
    },
    {
        "subject": "grammar and sentence structure",
        "construction": "provide one sentence and ask about a clearly identifiable grammatical role or correction",
        "evidence": "the supplied sentence",
        "context": "editing exercise",
    },
    {
        "subject": "vocabulary in context",
        "construction": "provide one sentence containing a target word and ask which meaning best fits its use",
        "evidence": "context clues in the supplied sentence",
        "context": "short reading excerpt",
    },
    {
        "subject": "reading comprehension",
        "construction": "provide a two-sentence passage and ask which conclusion is directly supported",
        "evidence": "the supplied passage only",
        "context": "brief informational paragraph",
    },
    {
        "subject": "logic and inference",
        "construction": "state two or three non-numeric facts and ask which conclusion must follow",
        "evidence": "the explicitly supplied facts",
        "context": "categorical reasoning",
    },
    {
        "subject": "technology and privacy literacy",
        "construction": "state a stable security rule in the scenario and ask which option follows that rule",
        "evidence": "the stated safety rule",
        "context": "account-security scenario",
    },
    {
        "subject": "scientific method",
        "construction": "describe an observation and controlled change and ask which statement identifies the testable variable or evidence",
        "evidence": "the experiment setup in the question",
        "context": "classroom observation",
    },
    {
        "subject": "categorical data interpretation",
        "construction": "provide a small list of labels or ordered categories and ask for a non-calculation interpretation",
        "evidence": "the listed labels or ordering",
        "context": "classification record",
    },
]

MCQ_GENERAL_STYLES = [
    "choose the statement directly supported by the supplied information",
    "identify the interpretation that follows from the supplied example",
    "select the only option consistent with the provided rule or passage",
]

MCQ_GENERAL_DISTRACTORS = [
    "Use plausible alternatives that are contradicted by a detail in the supplied evidence.",
    "Use nearby concepts, but ensure only one option follows from the supplied evidence.",
    "Use distinct choices and avoid trivia, opinion, or missing-context distractors.",
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

    if signal == "educational_qa_mcq_math":
        profile = _choose_profile(MCQ_MATH_VERIFIED_PROFILES, batch_id, signal, 1)
        difficulty = _choose(MCQ_MATH_DIFFICULTIES, batch_id, signal, 2)
        distractors = _choose(MCQ_MATH_DISTRACTOR_RULES, batch_id, signal, 3)
        answer_index = _choose_int(batch_id, signal, 4, 0, 3)

        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                "Verified mathematical MCQ mode: required.",
                f"Required family: {profile['family']}",
                f"Question construction: {profile['question_shape']}",
                f"Verification-expression shape: {profile['expression_shape']}",
                f"Context guidance: {profile['context']}",
                f"Difficulty guidance: {difficulty}",
                f"Correct-index target: {answer_index}",
                f"Distractor guidance: {distractors}",
                "The question must have one exact integer answer computable from the stated quantities.",
                "Return verification_expression and verification_answer for validator use; both must match a unique answer choice.",
                "Choices must be four distinct plain integer strings with no units.",
                "The explanation must show the numeric calculation and include the exact final integer answer.",
                "Do not generate next-step, best-explanation, conceptual, opinion, trivia, Python, or under-specified questions.",
                "Do not include the batch diversity id in any generated field.",
            ]
        )

    if signal == "educational_qa_mcq_general":
        profile = _choose_profile(MCQ_GENERAL_PROFILES, batch_id, signal, 1)
        style = _choose(MCQ_GENERAL_STYLES, batch_id, signal, 2)
        distractors = _choose(MCQ_GENERAL_DISTRACTORS, batch_id, signal, 3)
        answer_index = _choose_int(batch_id, signal, 4, 0, 3)

        return "\n".join(
            [
                f"Batch diversity id: {nonce}",
                "General non-math MCQ mode: required.",
                f"Subject focus: {profile['subject']}",
                f"Question construction: {profile['construction']}",
                f"Evidence requirement: {profile['evidence']}",
                f"Scenario context: {profile['context']}",
                f"Question style: {style}",
                f"Correct-index target: {answer_index}",
                f"Distractor guidance: {distractors}",
                "The correct answer must be justified entirely by information included in the question.",
                "Do not generate any arithmetic, fraction, percentage, ratio, geometry, probability, statistics, measurement, or financial calculation question.",
                "Do not generate trivia, current-fact recall, history/date recall, geography/location recall, next-step, opinion, or under-specified questions.",
                "Do not include the batch diversity id in any generated field.",
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
