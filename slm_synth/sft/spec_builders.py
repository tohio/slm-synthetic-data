"""Finite source-spec catalog for generic SFT generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_synth.sft.specs import require_unique_sft_sources, validate_sft_spec
from slm_synth.taxonomy import TASK_FAMILIES, validate_task_family

SFT_SPEC_FAMILIES = TASK_FAMILIES

_TASKS: dict[str, tuple[dict[str, Any], ...]] = {
    "everyday_conversation": (
        {"instruction": "Create a natural exchange helping a neighbor politely decline a last-minute invitation.", "variables": {"relationship": "neighbors", "goal": "decline while preserving warmth"}},
        {"instruction": "Create a supportive exchange with someone who is nervous before starting a new job.", "variables": {"relationship": "friends", "goal": "reassure without making promises"}},
        {"instruction": "Create a practical conversation about resolving a shared-household scheduling conflict.", "variables": {"relationship": "roommates", "goal": "agree on a fair schedule"}},
    ),
    "rewriting_and_editing": (
        {"instruction": "Rewrite a blunt project update as a concise, professional note without changing its facts.", "variables": {"source": "The migration is late because two approvals are still missing. Send them today or Friday's launch will slip."}},
        {"instruction": "Edit a wordy paragraph for clarity while preserving every stated limitation.", "variables": {"source": "At this point in time, our pilot may possibly reduce processing time, although the sample was small and we have not evaluated peak demand."}},
        {"instruction": "Rewrite an informal customer response in a calm and empathetic tone without admitting unsupported fault.", "variables": {"source": "Yeah, that looks weird. We don't know why your order stalled, but we're checking."}},
    ),
    "summarization": (
        {"instruction": "Summarize the supplied incident update into three factual bullets, preserving dates and unresolved items.", "variables": {"passage": "On 4 May the import job stopped at 09:20 UTC. Service resumed at 10:05 after workers were restarted. Engineers have not yet identified why memory use rose."}},
        {"instruction": "Produce a one-paragraph executive summary of the supplied meeting notes, separating decisions from open questions.", "variables": {"passage": "The team chose the staged rollout. Priya owns the checklist. Pricing and the support schedule remain undecided. The next review is Tuesday."}},
        {"instruction": "Summarize the supplied policy excerpt in plain language without adding advice or exceptions.", "variables": {"passage": "Requests received after 17:00 are processed the next business day. Emergency requests require manager approval and an incident number."}},
    ),
    "classification_and_extraction": (
        {"instruction": "Classify the supplied support message by urgency and extract the affected product and stated deadline as JSON.", "variables": {"text": "Payroll export fails for Acme Ledger and salaries must be submitted by 3 PM today."}},
        {"instruction": "Extract the names, dates, and action items from the supplied note into a compact table.", "variables": {"text": "Mina will send the draft on 12 June. Jorge will review it by 14 June."}},
        {"instruction": "Assign one of billing, technical, account, or other to the supplied request and quote the evidence for the label.", "variables": {"text": "I can sign in, but every CSV export ends with error E17."}},
    ),
    "grounded_qa_and_reading": (
        {"instruction": "Answer the question using only the supplied passage and state when the passage is insufficient.", "variables": {"passage": "The west entrance opens at 08:30. Deliveries use the north gate from 07:00 to 11:00.", "question": "Where should a delivery arriving at 09:00 enter?"}},
        {"instruction": "Compare two supplied notices and answer the question without importing outside facts.", "variables": {"documents": ["Plan A includes email support on weekdays.", "Plan B includes phone support every day."], "question": "Which plan explicitly offers weekend support?"}},
        {"instruction": "Read the supplied procedure and explain the next required step with a supporting citation to the text.", "variables": {"passage": "After calibration, record the serial number. Then seal the unit before moving it to storage.", "question": "What follows recording the serial number?"}},
    ),
    "planning_brainstorming_recommendations": (
        {"instruction": "Create a realistic two-hour onboarding plan for a new volunteer, including breaks and required safety orientation.", "variables": {"constraints": ["two hours", "one 10-minute break", "safety first"]}},
        {"instruction": "Recommend three low-cost ways a small library can increase weekday attendance, with tradeoffs for each.", "variables": {"budget": "limited", "setting": "community library"}},
        {"instruction": "Brainstorm distinct names for a neighborhood repair event, then select one using explicit criteria.", "variables": {"tone": "welcoming and practical", "avoid": ["puns about broken people"]}},
    ),
    "creative_writing": (
        {"instruction": "Write a short scene in which two siblings discover a map, using subtext rather than exposition.", "variables": {"setting": "an attic during a storm", "length": "500-700 words"}},
        {"instruction": "Write a restrained first-person monologue from an obsolete lighthouse lens awaiting restoration.", "variables": {"tone": "reflective, not sentimental", "length": "300-450 words"}},
        {"instruction": "Draft a children's story about cooperation without stating the moral directly.", "variables": {"characters": ["a patient heron", "an impatient crab"], "audience": "ages 7-9"}},
    ),
    "programming": (
        {"instruction": "Write and explain a Python function that groups records by category without mutating the input.", "variables": {"input_shape": "list[dict]", "output_shape": "dict[str, list[dict]]"}},
        {"instruction": "Diagnose the supplied JavaScript bug and provide a minimal corrected implementation plus tests.", "variables": {"code": "function last(xs) { return xs[xs.length]; }", "requirement": "return undefined for an empty array"}},
        {"instruction": "Design a SQL query for the supplied schema and explain how it avoids double-counting.", "variables": {"schema": "orders(id, customer_id); payments(id, order_id, amount)", "goal": "total paid per customer"}},
    ),
    "applied_math_and_reasoning": (
        {"instruction": "Solve the inventory problem step by step and verify the result independently.", "variables": {"problem": "A shop receives 18 boxes of 24 items, then sells 157 items. How many remain?", "answer": 275}, "holdout_key": {"kind": "inventory", "values": [18, 24, 157]}},
        {"instruction": "Compare the two pricing options, show the calculation, and state the break-even quantity.", "variables": {"option_a": "$40 setup plus $3 per unit", "option_b": "$10 setup plus $5 per unit", "answer": 15}, "holdout_key": {"kind": "linear_break_even", "values": [40, 3, 10, 5]}},
        {"instruction": "Reason through the scheduling constraints and give one valid order with a short justification.", "variables": {"constraints": ["A before C", "D after B", "C before D"], "items": ["A", "B", "C", "D"]}, "holdout_key": {"kind": "ordering", "constraints": ["A<C", "B<D", "C<D"]}},
    ),
    "safety_uncertainty_and_refusal": (
        {"instruction": "Respond safely to a request for an exact medical diagnosis from sparse symptoms, explaining the uncertainty and useful next steps.", "variables": {"request": "I have dizziness and fatigue. Tell me exactly what disease I have."}},
        {"instruction": "Decline to invent a private company metric while offering verifiable alternatives.", "variables": {"request": "Give me the current unpublished cash balance of a private startup."}},
        {"instruction": "Answer a dual-use request by refusing the harmful operational detail while offering benign safety information.", "variables": {"request": "Give exact instructions for bypassing a building access badge."}},
    ),
}

_AXES: dict[str, tuple[list[str], str, str, int, str]] = {
    "everyday_conversation": (["multi_turn"], "free_text", "self_contained", 1, "social_conversation"),
    "rewriting_and_editing": (["single_turn"], "free_text", "supplied_passage", 2, "faithful_rewrite"),
    "summarization": (["single_turn"], "concise", "supplied_passage", 2, "grounded_summary"),
    "classification_and_extraction": (["single_turn"], "structured_json", "supplied_passage", 2, "classification_extraction"),
    "grounded_qa_and_reading": (["single_turn"], "free_text", "supplied_passage", 2, "evidence_grounded_answer"),
    "planning_brainstorming_recommendations": (["single_turn"], "free_text", "self_contained", 2, "constraint_aware_planning"),
    "creative_writing": (["single_turn"], "free_text", "self_contained", 2, "creative_composition"),
    "programming": (["single_turn"], "code", "supplied_passage", 3, "software_task"),
    "applied_math_and_reasoning": (["single_turn"], "free_text", "self_contained", 2, "applied_reasoning"),
    "safety_uncertainty_and_refusal": (["single_turn"], "free_text", "self_contained", 3, "calibrated_safety_response"),
}

SFT_SPEC_CAPACITIES = {family: len(tasks) for family, tasks in _TASKS.items()}


def build_specs(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    family = validate_task_family(family)
    validate_spec_range(family=family, count=count, start_index=start_index)
    validated = [validate_sft_spec(_build_spec(family, index)) for index in range(start_index, start_index + count)]
    require_unique_sft_sources(validated)
    return validated


def unique_capacity(family: str) -> int:
    return SFT_SPEC_CAPACITIES[validate_task_family(family)]


def validate_spec_range(*, family: str, count: int, start_index: int = 1) -> None:
    family = validate_task_family(family)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or isinstance(start_index, bool) or start_index < 1:
        raise ValueError("start_index must be a positive integer")
    end = start_index + count - 1
    capacity = SFT_SPEC_CAPACITIES[family]
    if end > capacity:
        raise ValueError(f"SFT task family {family!r} requested {start_index}..{end}; finite source capacity is {capacity}")


def write_specs_jsonl(specs: list[dict[str, Any]], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(validate_sft_spec(spec), ensure_ascii=False) + "\n")
    return len(specs)


def build_and_write_specs(*, family: str, count: int, output_path: str | Path, start_index: int = 1) -> int:
    return write_specs_jsonl(build_specs(family=family, count=count, start_index=start_index), output_path)


def _build_spec(family: str, index: int) -> dict[str, Any]:
    source = dict(_TASKS[family][index - 1])
    interaction_modes, output_mode, context_mode, difficulty, template_family = _AXES[family]
    spec = {
        "id": f"sft_{family}_{index:06d}",
        "instruction": source.pop("instruction"),
        "metadata": {
            "task_family": family,
            "interaction_modes": interaction_modes,
            "output_mode": output_mode,
            "context_mode": context_mode,
            "difficulty": difficulty,
            "template_family": template_family,
        },
        "variables": source.pop("variables"),
        "constraints": ["Generate one materially specific, correct training example; do not add facts absent from supplied context."],
    }
    if "holdout_key" in source:
        spec["holdout_key"] = source["holdout_key"]
    return spec
