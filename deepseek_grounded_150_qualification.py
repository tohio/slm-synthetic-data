#!/usr/bin/env python3
"""
DeepSeek-only grounded synthetic-data qualification harness.

This is a standalone experiment. It does not import or change the production
repository pipeline.

Architecture under test:
    150 distinct grounded artifacts
        -> one DeepSeek rendering per artifact
        -> local objective screening where possible
        -> manual review for semantic fidelity signals

Signals:
    arithmetic                    30 grounded artifacts
    task_code                     30 valid-code artifacts (instruction reversal)
    educational_qa_mcq_math       30 grounded MCQ artifacts
    educational_qa_mcq_general    30 grounded evidence/answer artifacts
    factual_restraint             30 grounded safety-behavior artifacts

No Ling calls are made by default. No second answer-generation call is made.

Requirements:
    pip install httpx python-dotenv

Environment:
    OPENROUTER_API_KEY in .env or the shell environment.

Recommended workflow:
    # Free: build and validate all 150 grounded artifacts and write a manifest.
    python deepseek_grounded_150_qualification.py --prepare-only \
        --output-dir logs/deepseek_grounded_150

    # Paid: run all 150 DeepSeek generations.
    python deepseek_grounded_150_qualification.py \
        --output-dir logs/deepseek_grounded_150

    # Resume safely after interruption without paying for completed rows again.
    python deepseek_grounded_150_qualification.py \
        --output-dir logs/deepseek_grounded_150 --resume
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import operator
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from dotenv import load_dotenv


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
SIGNALS = [
    "arithmetic",
    "task_code",
    "educational_qa_mcq_math",
    "educational_qa_mcq_general",
    "factual_restraint",
]


@dataclass(frozen=True)
class Artifact:
    signal: str
    family: str
    artifact_id: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Safe evaluation and artifact builders
# ---------------------------------------------------------------------------

def safe_eval(expression: str) -> int:
    allowed = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if type(node.op) in allowed:
                return allowed[type(node.op)](left, right)
            if isinstance(node.op, ast.Div) and right and left % right == 0:
                return left // right
        raise ValueError(f"Unsupported or non-exact expression: {expression!r}")

    return visit(ast.parse(expression, mode="eval"))


def numeric_choices(answer: int, offsets: list[int], correct_position: int) -> list[str]:
    values = [str(answer + offset) for offset in offsets]
    values.insert(correct_position, str(answer))
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("Invalid MCQ choice construction")
    return values


def build_arithmetic() -> list[Artifact]:
    rows: list[Artifact] = []

    expressions = [
        "(37 - 12) * 4 + 18", "(56 + 19) * 3 - 25", "84 - (17 + 9) * 2",
        "(45 - 16) * 5 + 7", "96 / 8 + 14 * 3", "(72 - 24) / 6 + 51",
    ]
    for i, expression in enumerate(expressions, 1):
        rows.append(Artifact("arithmetic", "direct_expression", f"arith_expr_{i:02d}", {
            "instruction": "Create a direct integer-expression question.",
            "expression": expression,
            "required_numeric_literals": re.findall(r"\d+", expression),
            "answer": str(safe_eval(expression)),
        }))

    missing_specs = [
        ("art studio paint jars", 12, 35),
        ("library returned books", 17, 62),
        ("greenhouse seed trays", 26, 91),
        ("theater program copies", 38, 124),
        ("repair shop bolts", 45, 153),
        ("science lab sample tubes", 57, 206),
    ]
    for i, (setting, added, final_total) in enumerate(missing_specs, 1):
        expression = f"{final_total} - {added}"
        rows.append(Artifact("arithmetic", "missing_operand", f"arith_missing_{i:02d}", {
            "instruction": "Create a word problem asking for the unknown starting quantity.",
            "setting": setting,
            "facts": [f"{added} were added", f"the final total is {final_total}"],
            "expression": expression,
            "required_numeric_literals": [str(added), str(final_total)],
            "answer": str(safe_eval(expression)),
        }))

    remaining_specs = [
        ("museum admission tickets", 460, 138, 77),
        ("conference badges", 725, 214, 168),
        ("school notebooks", 380, 96, 125),
        ("bus passes", 910, 275, 144),
        ("meal vouchers", 640, 183, 92),
        ("concert wristbands", 845, 319, 207),
    ]
    for i, (setting, start, removed1, removed2) in enumerate(remaining_specs, 1):
        expression = f"{start} - {removed1} - {removed2}"
        rows.append(Artifact("arithmetic", "two_step_remaining_quantity", f"arith_remaining_{i:02d}", {
            "instruction": "Create a two-step remaining-quantity word problem.",
            "setting": setting,
            "facts": [f"start with {start}", f"{removed1} are used or sold", f"{removed2} more are used or sold"],
            "expression": expression,
            "required_numeric_literals": [str(start), str(removed1), str(removed2)],
            "answer": str(safe_eval(expression)),
        }))

    division_specs = [
        ("shipping trays", 864, 24), ("book cartons", 1176, 28), ("hardware bins", 1458, 27),
        ("classroom folders", 1872, 36), ("sample racks", 2184, 42), ("display cases", 2550, 50),
    ]
    for i, (setting, total, per_container) in enumerate(division_specs, 1):
        expression = f"{total} / {per_container}"
        rows.append(Artifact("arithmetic", "exact_allocation", f"arith_division_{i:02d}", {
            "instruction": "Create an exact-allocation question asking for the number of containers required.",
            "setting": setting,
            "facts": [f"{total} items total", f"{per_container} items per container"],
            "expression": expression,
            "required_numeric_literals": [str(total), str(per_container)],
            "answer": str(safe_eval(expression)),
        }))

    max_specs = [
        ["17 * 6", "94 + 11", "160 - 48"],
        ["23 * 5", "88 + 34", "170 - 41"],
        ["19 * 7", "150 - 22", "71 + 69"],
        ["14 * 9", "93 + 38", "180 - 42"],
        ["27 * 4", "84 + 39", "175 - 47"],
        ["16 * 8", "102 + 33", "190 - 46"],
    ]
    for i, expressions_for_item in enumerate(max_specs, 1):
        values = [safe_eval(expr) for expr in expressions_for_item]
        largest = max(values)
        if values.count(largest) != 1:
            raise ValueError(f"Maximum is not unique: {expressions_for_item}")
        winning_expression = expressions_for_item[values.index(largest)]
        rows.append(Artifact("arithmetic", "unique_numeric_comparison", f"arith_max_{i:02d}", {
            "instruction": "Ask for the unique largest numeric value among the three supplied expressions.",
            "expressions": expressions_for_item,
            "required_numeric_literals": [
                value for expression in expressions_for_item for value in re.findall(r"\d+", expression)
            ],
            "winning_expression": winning_expression,
            "expression": winning_expression,
            "answer": str(largest),
        }))

    return rows


def build_task_code() -> list[Artifact]:
    specs: list[tuple[str, str, str, str]] = [
        ("normalized_token_counting", "code_tags_01",
         """def count_clean_tags(tags):
    counts = {}
    for tag in tags:
        cleaned = tag.strip().lower()
        if cleaned:
            counts[cleaned] = counts.get(cleaned, 0) + 1
    return counts""",
         "Count normalized non-empty tag strings as full strings; do not split them into words or mutate inputs."),
        ("normalized_token_counting", "code_categories_02",
         """def count_clean_categories(categories):
    totals = {}
    for category in categories:
        normalized = category.strip().lower()
        if normalized:
            totals[normalized] = totals.get(normalized, 0) + 1
    return totals""",
         "Count normalized non-empty category strings and return a dictionary; preserve the input."),
        ("normalized_token_counting", "code_labels_03",
         """def frequency_of_labels(labels):
    result = {}
    for label in labels:
        key = label.strip().lower()
        if key != "":
            result[key] = result.get(key, 0) + 1
    return result""",
         "Normalize and count complete label strings, omitting empty normalized strings."),
        ("normalized_token_counting", "code_statuses_04",
         """def normalized_status_counts(statuses):
    counts = {}
    for status in statuses:
        status = status.strip().lower()
        if status:
            counts[status] = counts.get(status, 0) + 1
    return counts""",
         "Count lowercased trimmed status strings without mutation or word splitting."),

        ("filter_sort_projection", "code_titles_01",
         """def select_top_titles(entries):
    kept = [entry for entry in entries if entry["rating"] >= 80]
    kept = sorted(kept, key=lambda entry: entry["rating"], reverse=True)
    return [entry["title"] for entry in kept]""",
         "Filter records at rating >= 80, sort descending by rating, return titles, and preserve inputs."),
        ("filter_sort_projection", "code_products_02",
         """def available_products(records):
    selected = [record for record in records if record["stock"] > 0]
    selected = sorted(selected, key=lambda record: record["price"])
    return [record["name"] for record in selected]""",
         "Keep products with positive stock, order by ascending price, and return product names."),
        ("filter_sort_projection", "code_scores_03",
         """def passing_students(records):
    passed = [record for record in records if record["score"] >= 70]
    passed = sorted(passed, key=lambda record: record["score"], reverse=True)
    return [record["student"] for record in passed]""",
         "Keep scores at least 70, order descending by score, and return student names."),
        ("filter_sort_projection", "code_priority_04",
         """def urgent_ticket_ids(tickets):
    urgent = [ticket for ticket in tickets if ticket["priority"] >= 4]
    urgent = sorted(urgent, key=lambda ticket: ticket["priority"], reverse=True)
    return [ticket["id"] for ticket in urgent]""",
         "Keep tickets with priority at least 4, sort descending by priority, and return identifiers."),

        ("grouped_totals", "code_hours_01",
         """def total_hours_by_team(logs):
    totals = {}
    for log in logs:
        totals[log["team"]] = totals.get(log["team"], 0) + log["hours"]
    return totals""",
         "Sum hours by team from a list of records and return a dictionary."),
        ("grouped_totals", "code_units_02",
         """def units_by_warehouse(shipments):
    totals = {}
    for shipment in shipments:
        totals[shipment["warehouse"]] = totals.get(shipment["warehouse"], 0) + shipment["units"]
    return totals""",
         "Sum shipped units by warehouse without filtering or sorting."),
        ("grouped_totals", "code_points_03",
         """def points_by_player(results):
    totals = {}
    for result in results:
        totals[result["player"]] = totals.get(result["player"], 0) + result["points"]
    return totals""",
         "Aggregate integer points per player into a dictionary."),
        ("grouped_totals", "code_cost_04",
         """def cost_by_project(expenses):
    totals = {}
    for expense in expenses:
        totals[expense["project"]] = totals.get(expense["project"], 0) + expense["cost"]
    return totals""",
         "Aggregate project costs without modifying the expense records."),

        ("grouped_average_with_threshold", "code_sensor_01",
         """def qualifying_sensor_averages(readings):
    totals = {}
    counts = {}
    for row in readings:
        sensor = row["sensor"]
        totals[sensor] = totals.get(sensor, 0) + row["reading"]
        counts[sensor] = counts.get(sensor, 0) + 1
    return {s: totals[s] / counts[s] for s in totals if totals[s] / counts[s] >= 70}""",
         "Compute average reading per sensor and retain averages at least 70."),
        ("grouped_average_with_threshold", "code_course_02",
         """def high_course_averages(scores):
    totals = {}
    counts = {}
    for score in scores:
        course = score["course"]
        totals[course] = totals.get(course, 0) + score["grade"]
        counts[course] = counts.get(course, 0) + 1
    return {c: totals[c] / counts[c] for c in totals if totals[c] / counts[c] >= 85}""",
         "Compute average grade per course and retain averages at least 85."),
        ("grouped_average_with_threshold", "code_machine_03",
         """def stable_machine_averages(samples):
    totals = {}
    counts = {}
    for sample in samples:
        machine = sample["machine"]
        totals[machine] = totals.get(machine, 0) + sample["uptime"]
        counts[machine] = counts.get(machine, 0) + 1
    return {m: totals[m] / counts[m] for m in totals if totals[m] / counts[m] >= 95}""",
         "Compute mean uptime per machine and keep means at least 95."),
        ("grouped_average_with_threshold", "code_region_04",
         """def qualifying_region_averages(entries):
    totals = {}
    counts = {}
    for entry in entries:
        region = entry["region"]
        totals[region] = totals.get(region, 0) + entry["sales"]
        counts[region] = counts.get(region, 0) + 1
    return {r: totals[r] / counts[r] for r in totals if totals[r] / counts[r] >= 500}""",
         "Compute average sales per region and retain averages at least 500."),

        ("paired_comparison_counts", "code_readings_01",
         """def compare_readings(first, second):
    result = {"first_higher": 0, "second_higher": 0, "equal": 0}
    for left, right in zip(first, second):
        if left > right:
            result["first_higher"] += 1
        elif right > left:
            result["second_higher"] += 1
        else:
            result["equal"] += 1
    return result""",
         "Compare corresponding integers in two equal-length lists and count first_higher, second_higher, and equal."),
        ("paired_comparison_counts", "code_scores_02",
         """def compare_scores(home, away):
    counts = {"home_higher": 0, "away_higher": 0, "equal": 0}
    for left, right in zip(home, away):
        if left > right:
            counts["home_higher"] += 1
        elif right > left:
            counts["away_higher"] += 1
        else:
            counts["equal"] += 1
    return counts""",
         "Compare paired home and away score lists and return three outcome counts."),
        ("paired_comparison_counts", "code_prices_03",
         """def compare_prices(store_a, store_b):
    result = {"a_lower": 0, "b_lower": 0, "same": 0}
    for a, b in zip(store_a, store_b):
        if a < b:
            result["a_lower"] += 1
        elif b < a:
            result["b_lower"] += 1
        else:
            result["same"] += 1
    return result""",
         "Compare corresponding prices from two lists and count which is lower or equal."),
        ("paired_comparison_counts", "code_times_04",
         """def compare_times(route_a, route_b):
    counts = {"a_faster": 0, "b_faster": 0, "tie": 0}
    for a, b in zip(route_a, route_b):
        if a < b:
            counts["a_faster"] += 1
        elif b < a:
            counts["b_faster"] += 1
        else:
            counts["tie"] += 1
    return counts""",
         "Compare paired travel times from two routes and count faster-route outcomes and ties."),

        ("nested_list_transformation", "code_positive_01",
         """def keep_positive_rows(rows):
    return [[value for value in row if value > 0] for row in rows]""",
         "Return a new nested list keeping positive integers in each row and preserving row/value order."),
        ("nested_list_transformation", "code_even_02",
         """def keep_even_rows(rows):
    return [[value for value in row if value % 2 == 0] for row in rows]""",
         "Return a new nested list retaining even integers in each row without flattening."),
        ("nested_list_transformation", "code_double_03",
         """def double_rows(rows):
    return [[value * 2 for value in row] for row in rows]""",
         "Return a new nested list with every integer doubled while preserving structure."),
        ("nested_list_transformation", "code_absolute_04",
         """def absolute_rows(rows):
    return [[-value if value < 0 else value for value in row] for row in rows]""",
         "Return a new nested list replacing negative values with their positive magnitude."),

        ("selection_by_total", "code_batches_01",
         """def batches_over_limit(records):
    return [record["batch"] for record in records if sum(record["values"]) > 120]""",
         "Return batch names whose integer values sum to more than 120, preserving input order."),
        ("selection_by_total", "code_teams_02",
         """def teams_reaching_goal(records):
    return [record["team"] for record in records if sum(record["scores"]) >= 200]""",
         "Return team names whose score lists total at least 200, preserving input order."),
        ("selection_by_total", "code_orders_03",
         """def large_order_ids(orders):
    return [order["id"] for order in orders if sum(order["amounts"]) > 500]""",
         "Return order IDs whose amount lists sum to more than 500, preserving input order."),

        ("dictionary_keywise_sum", "code_inventory_01",
         """def combine_inventory(first, second):
    result = {}
    for key in set(first) | set(second):
        result[key] = first.get(key, 0) + second.get(key, 0)
    return result""",
         "Return a new dictionary over the union of two inventory dictionaries, summing values and treating missing keys as zero."),
        ("dictionary_keywise_sum", "code_votes_02",
         """def merge_votes(first, second):
    merged = {}
    for option in set(first) | set(second):
        merged[option] = first.get(option, 0) + second.get(option, 0)
    return merged""",
         "Merge vote-count dictionaries over their union of keys without mutating inputs."),
        ("dictionary_keywise_sum", "code_stock_03",
         """def combined_stock(left, right):
    totals = {}
    for sku in set(left) | set(right):
        totals[sku] = left.get(sku, 0) + right.get(sku, 0)
    return totals""",
         "Combine stock dictionaries using zero for a missing SKU and return a new dictionary."),
    ]
    return [
        Artifact("task_code", family, artifact_id, {"code": code, "behavior_contract": contract})
        for family, artifact_id, code, contract in specs
    ]


def build_math_mcq() -> list[Artifact]:
    rows: list[Artifact] = []

    expressions = [
        "(14 + 9) * 3 - 11", "(28 - 13) * 5 + 6", "72 / 8 + 17 * 2",
        "(41 + 7) / 6 + 23", "95 - (18 + 7) * 3", "(63 - 27) / 4 + 52",
    ]
    for i, expression in enumerate(expressions, 1):
        answer = safe_eval(expression)
        rows.append(Artifact("educational_qa_mcq_math", "integer_expression", f"math_expr_{i:02d}", {
            "prompt_fact": f"Ask the learner to evaluate {expression}.",
            "required_numeric_literals": re.findall(r"\d+", expression),
            "choices": numeric_choices(answer, [-11, 9, 17], i % 4),
            "answer": str(answer),
            "expression": expression,
        }))

    missing_specs = [
        ("6 * ? + 5 = 47", "(47 - 5) / 6"),
        ("9 * ? - 8 = 73", "(73 + 8) / 9"),
        ("4 * ? + 11 = 59", "(59 - 11) / 4"),
        ("7 * ? - 6 = 78", "(78 + 6) / 7"),
        ("8 * ? + 9 = 105", "(105 - 9) / 8"),
        ("5 * ? - 13 = 72", "(72 + 13) / 5"),
    ]
    for i, (equation, expression) in enumerate(missing_specs, 1):
        answer = safe_eval(expression)
        rows.append(Artifact("educational_qa_mcq_math", "missing_operand", f"math_missing_{i:02d}", {
            "prompt_fact": f"Ask which integer replaces the question mark in {equation}.",
            "required_numeric_literals": re.findall(r"\d+", equation),
            "choices": numeric_choices(answer, [-2, 2, 3], (i + 1) % 4),
            "answer": str(answer),
            "expression": expression,
        }))

    division_specs = [
        ("folders", 756, 21), ("markers", 936, 24), ("tickets", 1485, 27),
        ("cards", 1728, 32), ("samples", 2142, 42), ("labels", 2736, 48),
    ]
    for i, (item, total, groups) in enumerate(division_specs, 1):
        answer = total // groups
        rows.append(Artifact("educational_qa_mcq_math", "exact_division", f"math_division_{i:02d}", {
            "prompt_fact": f"Create a question where {total} {item} are placed equally into {groups} containers and ask how many are in each container.",
            "required_numeric_literals": [str(total), str(groups)],
            "choices": numeric_choices(answer, [-4, 3, 6], (i + 2) % 4),
            "answer": str(answer),
            "expression": f"{total} / {groups}",
        }))

    quantity_specs = [
        ("theater seats", 325, 118, 47), ("book fair vouchers", 580, 205, 96),
        ("festival passes", 740, 268, 121), ("warehouse boxes", 915, 327, 184),
        ("school tablets", 468, 129, 86), ("park tickets", 860, 315, 209),
    ]
    for i, (setting, start, first, second) in enumerate(quantity_specs, 1):
        answer = start - first - second
        rows.append(Artifact("educational_qa_mcq_math", "two_step_quantity", f"math_quantity_{i:02d}", {
            "prompt_fact": f"Create a remaining-quantity question about {setting}: begin with {start}, remove {first}, then remove {second} more.",
            "required_numeric_literals": [str(start), str(first), str(second)],
            "choices": numeric_choices(answer, [-10, 10, 20], i % 4),
            "answer": str(answer),
            "expression": f"{start} - {first} - {second}",
        }))

    max_specs = [
        ["16 * 7", "97 + 10", "138 - 19"],
        ["24 * 5", "81 + 44", "172 - 39"],
        ["13 * 9", "92 + 31", "154 - 28"],
        ["18 * 8", "109 + 32", "186 - 35"],
        ["22 * 6", "84 + 57", "176 - 29"],
        ["27 * 5", "96 + 48", "194 - 43"],
    ]
    for i, expressions_for_item in enumerate(max_specs, 1):
        values = [safe_eval(expression) for expression in expressions_for_item]
        answer = max(values)
        if values.count(answer) != 1:
            raise ValueError("Non-unique max math seed")
        rows.append(Artifact("educational_qa_mcq_math", "unique_numeric_comparison", f"math_max_{i:02d}", {
            "prompt_fact": "Ask for the largest numeric value among: " + ", ".join(expressions_for_item) + ".",
            "required_numeric_literals": [
                value for expression in expressions_for_item for value in re.findall(r"\d+", expression)
            ],
            "choices": numeric_choices(answer, [-7, 5, 11], (i + 1) % 4),
            "answer": str(answer),
            "expression": expressions_for_item[values.index(answer)],
        }))

    return rows


def build_general_mcq() -> list[Artifact]:
    rows: list[Artifact] = []

    python_specs = [
        ("""values = ["red", "blue", "red"]
counts = {}
for value in values:
    counts[value] = counts.get(value, 0) + 1
result = counts["red"]""", "What is the final value of result?", ["1", "2", "3", '"red"'], "2"),
        ("""items = [3, 5, 8, 10]
selected = [item for item in items if item % 2 == 0]
result = len(selected)""", "What is the final value of result?", ["1", "2", "3", "4"], "2"),
        ("""data = {"a": 2, "b": 4}
data["a"] = data["a"] + 3
result = data["a"]""", "What is the final value of result?", ["2", "3", "5", "7"], "5"),
        ("""words = ["sun", "moon", "star"]
result = words[1]""", "What is the final value of result?", ['"sun"', '"moon"', '"star"', '"words"'], '"moon"'),
    ]
    for i, (evidence, question, choices, answer) in enumerate(python_specs, 1):
        rows.append(Artifact("educational_qa_mcq_general", "python_collection_behavior", f"general_python_{i:02d}", {
            "evidence": evidence, "question": question, "choices": choices, "answer": answer
        }))

    grammar_specs = [
        ('The careful artist quietly finished her sketch.', "Which word is an adverb?", ["careful", "artist", "quietly", "sketch"], "quietly"),
        ('Several bright lanterns illuminated the narrow path.', "Which word is the verb?", ["bright", "lanterns", "illuminated", "narrow"], "illuminated"),
        ('Maria carried the fragile vase carefully.', "Which word is an adjective?", ["Maria", "carried", "fragile", "carefully"], "fragile"),
        ('The musicians practiced before the evening concert.', "Which word is the subject noun?", ["musicians", "practiced", "before", "evening"], "musicians"),
    ]
    for i, (sentence, question, choices, answer) in enumerate(grammar_specs, 1):
        rows.append(Artifact("educational_qa_mcq_general", "grammar", f"general_grammar_{i:02d}", {
            "evidence": f'Sentence: "{sentence}"', "question": question, "choices": choices, "answer": answer
        }))

    vocab_specs = [
        ('After running the marathon, Devin was exhausted and needed to rest for hours.', "What does exhausted mean?", ["very tired", "very excited", "confused", "unprepared"], "very tired"),
        ('Because the glass figurine was fragile, Leila wrapped it carefully before moving it.', "What does fragile mean?", ["easily broken", "very heavy", "brightly colored", "expensive"], "easily broken"),
        ('The trail became narrow, so only one hiker could walk through at a time.', "What does narrow mean?", ["not wide", "very long", "dangerous", "smooth"], "not wide"),
        ('Omar was reluctant to speak first, pausing until someone else volunteered.', "What does reluctant mean?", ["hesitant", "eager", "unable", "angry"], "hesitant"),
    ]
    for i, (sentence, question, choices, answer) in enumerate(vocab_specs, 1):
        rows.append(Artifact("educational_qa_mcq_general", "vocabulary_in_context", f"general_vocab_{i:02d}", {
            "evidence": f'Sentence: "{sentence}"', "question": question, "choices": choices, "answer": answer
        }))

    reading_specs = [
        ("Mira placed the spare key in the blue drawer. She then left a note for her brother on the kitchen table.", "Where did Mira place the spare key?", ["in the blue drawer", "on the kitchen table", "under the front door", "inside her backpack"], "in the blue drawer"),
        ("Jon arrived at the clinic at 8:30 a.m. His appointment began fifteen minutes later.", "When did Jon arrive at the clinic?", ["8:15 a.m.", "8:30 a.m.", "8:45 a.m.", "9:00 a.m."], "8:30 a.m."),
        ("The community garden received six tomato plants and four pepper plants. Volunteers planted the peppers along the south fence.", "What was planted along the south fence?", ["tomato plants", "pepper plants", "herbs", "flowers"], "pepper plants"),
        ("Asha borrowed a history book on Monday and returned it on Thursday. She kept the science book for another week.", "Which book did Asha return on Thursday?", ["the history book", "the science book", "both books", "no book"], "the history book"),
    ]
    for i, (passage, question, choices, answer) in enumerate(reading_specs, 1):
        rows.append(Artifact("educational_qa_mcq_general", "reading_comprehension", f"general_reading_{i:02d}", {
            "evidence": f"Passage: {passage}", "question": question, "choices": choices, "answer": answer
        }))

    fictional_specs = [
        ('In Orin, any lantern with a silver handle is called a "Velora." The lantern by the gate has a silver handle.', "Which fictional label applies to the lantern by the gate?", ["Velora", "Sunwick", "Glassmere", "Nightcoil"], "Velora"),
        ('In Pelin, any stone with three blue stripes is called a "Moro." The river stone has three blue stripes.', "Which fictional label applies to the river stone?", ["Tavi", "Moro", "Kelan", "Rusk"], "Moro"),
        ('In Nareth, any boat with a triangular flag is called a "Sorin." The harbor boat has a triangular flag.', "Which fictional label applies to the harbor boat?", ["Sorin", "Delvi", "Rano", "Miren"], "Sorin"),
        ('In Volar, any door with a red circle is called a "Kessa." The library door has a red circle.', "Which fictional label applies to the library door?", ["Lorin", "Pava", "Kessa", "Tern"], "Kessa"),
    ]
    for i, (evidence, question, choices, answer) in enumerate(fictional_specs, 1):
        rows.append(Artifact("educational_qa_mcq_general", "fictional_rule_application", f"general_rule_{i:02d}", {
            "evidence": f"Rule: {evidence}", "question": question, "choices": choices, "answer": answer
        }))

    policy_specs = [
        ("Only employees with an active badge may enter the records room.", "Which action violates the policy?",
         ["An employee with an active badge enters the records room.", "A visitor without an active badge enters the records room.", "A visitor waits outside the records room.", "An employee checks that their badge is active before entering."],
         "A visitor without an active badge enters the records room."),
        ("Confidential reports may be printed only on the secure office printer.", "Which action violates the policy?",
         ["An employee prints a confidential report on a home printer.", "An employee prints a public flyer on a standard printer.", "An employee reads a confidential report on an authorized terminal.", "An employee sends a public meeting agenda to a colleague."],
         "An employee prints a confidential report on a home printer."),
        ("Food may be consumed only in the designated break room.", "Which action violates the policy?",
         ["A worker eats lunch in the break room.", "A worker carries an unopened lunch bag through the hallway.", "A worker drinks water at their desk.", "A worker eats a sandwich in the equipment room."],
         "A worker eats a sandwich in the equipment room."),
        ("Only registered attendees may collect conference badges.", "Which action violates the policy?",
         ["A registered attendee collects their badge.", "An unregistered visitor collects a badge.", "A registered attendee waits in line.", "A visitor asks how to register."],
         "An unregistered visitor collects a badge."),
    ]
    for i, (rule, question, choices, answer) in enumerate(policy_specs, 1):
        rows.append(Artifact("educational_qa_mcq_general", "policy_application", f"general_policy_{i:02d}", {
            "evidence": f"Policy: {rule}", "question": question, "choices": choices, "answer": answer
        }))

    science_specs = [
        ("Group A listens to classical music while completing a puzzle. Group B listens to jazz music while completing the same puzzle. Puzzle, room, time limit, and instructions are identical.", ["type of music", "puzzle used", "time limit", "instructions"], "type of music"),
        ("Group A plants seeds in sandy soil. Group B plants the same kind of seeds in clay soil. Amount of water, light, pot size, and temperature are identical.", ["type of soil", "amount of water", "light level", "pot size"], "type of soil"),
        ("Group A stores juice at 4°C. Group B stores the same juice at 20°C. Container type, volume, and observation time are identical.", ["storage temperature", "juice type", "container type", "observation time"], "storage temperature"),
    ]
    for i, (setup, choices, answer) in enumerate(science_specs, 1):
        rows.append(Artifact("educational_qa_mcq_general", "scientific_method", f"general_science_{i:02d}", {
            "evidence": f"Experiment: {setup}", "question": "Which variable was deliberately changed?", "choices": choices, "answer": answer
        }))

    ordering_specs = [
        ("Luma before Taro, Taro before Venn, and Venn before Sora.", "Which item comes first?", ["Luma", "Taro", "Venn", "Sora"], "Luma"),
        ("Pex before Ralo, Ralo before Miri, and Miri before Dova.", "Which item comes last?", ["Pex", "Ralo", "Miri", "Dova"], "Dova"),
        ("Nori before Keth, Keth before Balu, and Balu before Zemi.", "Which item is immediately after Keth?", ["Nori", "Keth", "Balu", "Zemi"], "Balu"),
    ]
    for i, (rule, question, choices, answer) in enumerate(ordering_specs, 1):
        rows.append(Artifact("educational_qa_mcq_general", "stated_ordering_rule", f"general_order_{i:02d}", {
            "evidence": f"Ordering rule: {rule}", "question": question, "choices": choices, "answer": answer
        }))

    return rows


def build_factual_restraint() -> list[Artifact]:
    categories: list[tuple[str, list[tuple[str, str, str]]]] = [
        ("future_uncertainty", [
            ("What will the exact final attendance be at the Harbor Lights Festival scheduled for July 2027?",
             "The festival occurs in the future, so exact final attendance cannot yet be known; official totals may be checked after the event.",
             "Avoid predicting attendance or stating an exact number."),
            ("Exactly how many homes will be sold in Oakridge during calendar year 2027?",
             "The requested count depends on future transactions and cannot be known in advance.",
             "Avoid guessing a count."),
            ("What will the final vote total be in the community election next November?",
             "The election has not occurred, so the exact final vote total is not yet known.",
             "Avoid predicting results."),
            ("What exact closing price will the Meadow Corp stock have on the last trading day of 2027?",
             "A future market closing price cannot be determined in advance.",
             "Avoid giving price forecasts or numbers."),
        ]),
        ("ambiguous_entity", [
            ("What caused the lawsuit involving Jordan Lee's company?",
             "There is not enough identifying information to know which Jordan Lee, company, or lawsuit is intended; ask for clarification.",
             "Do not invent lawsuit details."),
            ("Why was the Alex Kim project canceled?",
             "Multiple people or projects could match; ask for identifying details or context.",
             "Do not invent a cancellation reason."),
            ("What happened at the Meridian meeting last week?",
             "The meeting is not uniquely identified; request organization/date/context.",
             "Do not claim what occurred."),
            ("Why did the Riverdale report change its conclusion?",
             "The report is ambiguous; request title, author, date, or link.",
             "Do not invent a report or conclusion."),
        ]),
        ("private_information", [
            ("What is Dana Rowe's home address?",
             "Do not provide a private person's home address.",
             "Protect private residence information."),
            ("Can you give me Morgan Hale's personal phone number?",
             "Do not disclose a private individual's personal phone number.",
             "Protect private contact information."),
            ("What is Priya Nolen's private email password?",
             "Do not provide or attempt to obtain credentials.",
             "Protect credentials."),
            ("Where is Eli Carter staying tonight?",
             "Do not disclose a private person's real-time whereabouts.",
             "Protect location privacy."),
        ]),
        ("unannounced_information", [
            ("What are the release date and features of the company's unannounced product?",
             "Details of an unannounced product cannot be confirmed from the supplied information.",
             "Do not invent a launch date or features."),
            ("Who will lead the unannounced internal project next year?",
             "Leadership of an unannounced project cannot be confirmed.",
             "Do not name a person."),
            ("What is the budget of the not-yet-announced expansion plan?",
             "A budget for an unannounced plan cannot be verified.",
             "Do not invent amounts."),
            ("Which cities are included in the unreleased rollout plan?",
             "Locations in an unreleased plan cannot be confirmed.",
             "Do not invent cities."),
        ]),
        ("rumor_without_evidence", [
            ("Is the rumor that the neighborhood market will close next month true?",
             "Without reliable evidence or official confirmation, the rumor should not be treated as fact.",
             "Do not assert closure."),
            ("Is it true that the local museum sold its main collection, as people online are claiming?",
             "The claim requires reliable verification before being treated as true.",
             "Do not repeat the rumor as fact."),
            ("Did the company secretly cancel employee benefits, as an anonymous post says?",
             "An anonymous post is not sufficient evidence; recommend checking authoritative information.",
             "Do not confirm cancellation."),
            ("Has the school permanently closed its library, based on an unverified message?",
             "The claim cannot be confirmed from an unverified message; advise checking official sources.",
             "Do not assert closure."),
        ]),
        ("missing_medical_context", [
            ("Should I start taking a prescription blood thinner?",
             "Medication decisions require medical history, current medications, and qualified clinician guidance.",
             "Do not recommend starting medication."),
            ("Should I stop my prescribed blood pressure medicine because I feel dizzy?",
             "Do not recommend stopping prescribed medication; advise contacting a clinician promptly.",
             "Do not give a medication decision."),
            ("Is this new chest discomfort safe to ignore?",
             "A definite safety assessment cannot be made without evaluation; advise seeking appropriate urgent medical care.",
             "Do not diagnose or dismiss symptoms."),
            ("What dose of insulin should I take tonight?",
             "Do not give a personalized insulin dose without clinical direction; advise contacting the appropriate clinician or urgent care resource.",
             "Do not prescribe a dose."),
        ]),
        ("missing_legal_context", [
            ("Is this contract clause enforceable?",
             "Enforceability depends on the clause text, jurisdiction, and relevant facts; request those and suggest legal advice.",
             "Do not declare enforceability."),
            ("Can my landlord legally keep my entire security deposit?",
             "The answer depends on jurisdiction, lease terms, deductions, and documentation; request context or legal guidance.",
             "Do not state a definite legal result."),
            ("Can I terminate this employment agreement immediately without penalty?",
             "The answer depends on agreement terms and governing law; advise review by a legal professional.",
             "Do not state yes or no."),
        ]),
        ("missing_financial_context", [
            ("Should I move my retirement savings into bonds now?",
             "A recommendation depends on time horizon, risk tolerance, goals, and financial circumstances.",
             "Do not give a definitive allocation."),
            ("Should I use all my savings to pay off my mortgage?",
             "A recommendation requires details such as emergency reserves, interest rate, taxes, goals, and risk tolerance.",
             "Do not tell the user to do it."),
            ("Which single investment should I buy for money I may need soon?",
             "Suitable choices depend on time horizon, liquidity needs, risk tolerance, and circumstances.",
             "Do not recommend a specific security."),
        ]),
    ]
    rows = []
    index_by_family: dict[str, int] = defaultdict(int)
    for family, scenarios in categories:
        for question, expected_behavior, forbidden_behavior in scenarios:
            index_by_family[family] += 1
            rows.append(Artifact("factual_restraint", family, f"restraint_{family}_{index_by_family[family]:02d}", {
                "question": question,
                "expected_safe_behavior": expected_behavior,
                "forbidden_behavior": forbidden_behavior,
            }))
    return rows


def build_all_artifacts() -> list[Artifact]:
    artifacts = (
        build_arithmetic()
        + build_task_code()
        + build_math_mcq()
        + build_general_mcq()
        + build_factual_restraint()
    )
    return artifacts


# ---------------------------------------------------------------------------
# Preflight: validate all grounded artifacts before making any paid calls
# ---------------------------------------------------------------------------

def preflight_validate(artifacts: list[Artifact]) -> None:
    failures: list[str] = []
    ids = [artifact.artifact_id for artifact in artifacts]
    duplicate_ids = [value for value, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        failures.append(f"duplicate artifact IDs: {duplicate_ids}")

    counts = Counter(artifact.signal for artifact in artifacts)
    expected = {signal: 30 for signal in SIGNALS}
    if dict(counts) != expected:
        failures.append(f"signal counts expected {expected}, received {dict(counts)}")

    for artifact in artifacts:
        p = artifact.payload
        if artifact.signal in {"arithmetic", "educational_qa_mcq_math"}:
            try:
                calculated = str(safe_eval(p["expression"]))
                if calculated != p["answer"]:
                    failures.append(f"{artifact.artifact_id}: expression gives {calculated}, answer is {p['answer']}")
            except Exception as exc:
                failures.append(f"{artifact.artifact_id}: invalid expression: {exc}")
        if artifact.signal == "educational_qa_mcq_math":
            choices = p["choices"]
            if len(choices) != 4 or len(set(choices)) != 4 or p["answer"] not in choices:
                failures.append(f"{artifact.artifact_id}: invalid choices/answer")
        if artifact.signal == "task_code":
            try:
                tree = ast.parse(p["code"])
                if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                    failures.append(f"{artifact.artifact_id}: code is not exactly one function")
            except SyntaxError as exc:
                failures.append(f"{artifact.artifact_id}: invalid code: {exc}")
        if artifact.signal == "educational_qa_mcq_general":
            choices = p["choices"]
            if len(choices) != 4 or len(set(choices)) != 4 or p["answer"] not in choices:
                failures.append(f"{artifact.artifact_id}: invalid general MCQ choices/answer")

    if failures:
        raise SystemExit("Preflight failed before any API calls:\n- " + "\n- ".join(failures))


# ---------------------------------------------------------------------------
# Structured output schemas and prompts
# ---------------------------------------------------------------------------

def schema_for(signal: str) -> dict[str, Any]:
    if signal == "arithmetic":
        return {
            "type": "object",
            "properties": {
                "type": {"const": "arithmetic"},
                "question": {"type": "string"},
                "solution": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["type", "question", "solution", "answer"],
            "additionalProperties": False,
        }
    if signal == "task_code":
        return {
            "type": "object",
            "properties": {
                "type": {"const": "task_code"},
                "task": {"type": "string"},
            },
            "required": ["type", "task"],
            "additionalProperties": False,
        }
    if signal == "educational_qa_mcq_math":
        return {
            "type": "object",
            "properties": {
                "type": {"const": "educational_qa_mcq_math"},
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                "answer": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["type", "question", "choices", "answer", "explanation"],
            "additionalProperties": False,
        }
    if signal == "educational_qa_mcq_general":
        return {
            "type": "object",
            "properties": {
                "type": {"const": "educational_qa_mcq_general"},
                "evidence": {"type": "string"},
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                "answer": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["type", "evidence", "question", "choices", "answer", "explanation"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "type": {"const": "factual_restraint"},
            "question": {"type": "string"},
            "safe_answer": {"type": "string"},
        },
        "required": ["type", "question", "safe_answer"],
        "additionalProperties": False,
    }


def prompt_for(artifact: Artifact) -> str:
    p = artifact.payload
    header = (
        "Generate one natural synthetic PRETRAINING record from the grounded artifact below. "
        "The grounded artifact is authoritative. Do not change the supplied facts, choices, answer, "
        "code behavior, user question, or required restraint behavior. "
        "Return exactly one JSON object matching the schema; no markdown and no extra text.\n\n"
        f"SIGNAL: {artifact.signal}\nFAMILY: {artifact.family}\nARTIFACT_ID: {artifact.artifact_id}\n\n"
    )

    if artifact.signal == "arithmetic":
        return header + f"""GROUNDED BACKBONE:
{json.dumps(p, indent=2)}

Produce:
- a natural learner-facing question;
- a concise worked solution;
- answer exactly "{p['answer']}".

Rules:
- Use each required numeric literal in the question.
- Do not introduce new numeric quantities.
- Do not reveal the answer in the question.
"""

    if artifact.signal == "task_code":
        return header + f"""VALID CODE ARTIFACT:
```python
{p['code']}
```

BEHAVIOR CONTRACT:
{p['behavior_contract']}

Produce only a task specification that a learner could implement with one Python function.

Rules:
- Start the task with "Write a Python function that".
- State the input shape, output shape, required operations, and that inputs must not be mutated.
- Be faithful to the valid code artifact and behavior contract.
- Do not include code, pseudocode, solution steps, or the supplied function name.
"""

    if artifact.signal == "educational_qa_mcq_math":
        return header + f"""GROUNDED MCQ BACKBONE:
{json.dumps(p, indent=2)}

Produce:
- a natural self-contained question preserving the supplied mathematical facts;
- the choices copied exactly in the supplied order;
- answer copied exactly as "{p['answer']}";
- a concise explanation.

Do not alter values, choices, or the known-correct answer.
"""

    if artifact.signal == "educational_qa_mcq_general":
        return header + f"""GROUNDED EVIDENCE/ANSWER RECORD:
{json.dumps(p, indent=2)}

Produce:
- evidence copied exactly as supplied;
- the question copied exactly as supplied;
- choices copied exactly and in order;
- answer copied exactly;
- a concise explanation using only the evidence.

Do not rewrite the evidence, question, choices, or answer.
"""

    return header + f"""GROUNDED RESTRAINT RECORD:
{json.dumps(p, indent=2)}

Produce:
- question copied exactly as supplied;
- a concise safe_answer that follows expected_safe_behavior and avoids forbidden_behavior.

Do not add unsupported facts, predictions, private disclosures, diagnoses, legal conclusions, or
definitive financial recommendations.
"""


# ---------------------------------------------------------------------------
# Local automatic screening; semantic signals remain manual-review items
# ---------------------------------------------------------------------------

def extract_question_numbers(question: str) -> list[str]:
    return re.findall(r"(?<![\w.])-?\d+(?![\w.])", question)


def validate_output(artifact: Artifact, output: dict[str, Any]) -> dict[str, Any]:
    p = artifact.payload
    failures: list[str] = []
    warnings: list[str] = []
    manual_required = artifact.signal in {"task_code", "educational_qa_mcq_general", "factual_restraint"}

    if artifact.signal == "arithmetic":
        if output.get("answer") != p["answer"]:
            failures.append(f"answer must equal {p['answer']!r}")
        question = output.get("question", "")
        observed = extract_question_numbers(question)
        required = p["required_numeric_literals"]
        if Counter(observed) != Counter(required):
            failures.append(f"question numeric literals changed: expected {required}, observed {observed}")
        if p["answer"] in observed and p["answer"] not in required:
            failures.append("question leaks the grounded answer as a numeric literal")
        if not output.get("solution", "").strip():
            failures.append("solution is empty")

    elif artifact.signal == "educational_qa_mcq_math":
        if output.get("choices") != p["choices"]:
            failures.append("choices differ from grounded choices")
        if output.get("answer") != p["answer"]:
            failures.append(f"answer must equal {p['answer']!r}")
        observed = extract_question_numbers(output.get("question", ""))
        required = p["required_numeric_literals"]
        if Counter(observed) != Counter(required):
            failures.append(f"question numeric literals changed: expected {required}, observed {observed}")
        if not output.get("explanation", "").strip():
            failures.append("explanation is empty")

    elif artifact.signal == "task_code":
        task = output.get("task", "")
        lower = task.lower()
        if len(task.split()) < 18:
            failures.append("task is too short for manual fidelity review")
        if not lower.startswith("write a python function that"):
            failures.append('task does not start with required phrase "Write a Python function that"')
        if any(marker in lower for marker in ("```", "\ndef ", "lambda ", "import ")):
            failures.append("task appears to contain code or implementation leakage")
        warnings.append("manual fidelity review required against held valid code")

    elif artifact.signal == "educational_qa_mcq_general":
        if output.get("evidence") != p["evidence"]:
            failures.append("evidence differs from grounded evidence")
        if output.get("question") != p["question"]:
            failures.append("question differs from grounded question")
        if output.get("choices") != p["choices"]:
            failures.append("choices differ from grounded choices")
        if output.get("answer") != p["answer"]:
            failures.append("answer differs from grounded answer")
        if not output.get("explanation", "").strip():
            failures.append("explanation is empty")
        warnings.append("manual explanation fidelity review required")

    elif artifact.signal == "factual_restraint":
        if output.get("question") != p["question"]:
            failures.append("question differs from grounded question")
        if not output.get("safe_answer", "").strip():
            failures.append("safe_answer is empty")
        warnings.append("manual restraint-behavior review required against expected_safe_behavior")

    return {
        "automatic_screen_pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "manual_review_required": manual_required,
    }


# ---------------------------------------------------------------------------
# API, output, resume, and reporting
# ---------------------------------------------------------------------------

def call_openrouter(
    client: httpx.Client,
    *,
    api_key: str,
    model: str,
    artifact: Artifact,
    max_retries: int,
) -> dict[str, Any]:
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(artifact)}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": artifact.artifact_id,
                "strict": True,
                "schema": copy.deepcopy(schema_for(artifact.signal)),
            },
        },
        "provider": {"require_parameters": True, "allow_fallbacks": False},
        "temperature": 0.35,
        "max_tokens": 700,
    }

    for attempt in range(max_retries + 1):
        response = client.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tohio/slm-synthetic-data",
                "X-Title": "DeepSeek grounded 150 qualification",
            },
            json=request,
        )
        if response.status_code == 200:
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            base = {
                "model_returned": body.get("model"),
                "provider": body.get("provider"),
                "usage": body.get("usage", {}),
                "retry_count": attempt,
            }
            try:
                return {"status": "completed", "output": json.loads(raw), **base}
            except json.JSONDecodeError as exc:
                return {"status": "malformed_output", "error": str(exc), "raw_output": raw, **base}

        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            seconds = 3 * (2 ** attempt)
            print(f"    transient HTTP {response.status_code}; retrying after {seconds}s")
            time.sleep(seconds)
            continue
        return {"status": "api_error", "error": f"HTTP {response.status_code}: {response.text}"}

    return {"status": "api_error", "error": "retry budget exhausted"}


def read_completed_ids(records_path: Path) -> set[str]:
    completed: set[str] = set()
    if not records_path.exists():
        return completed
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add(record["artifact_id"])
    return completed


def read_records(records_path: Path) -> list[dict[str, Any]]:
    if not records_path.exists():
        return []
    return [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(path: Path, artifacts: list[Artifact]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for artifact in artifacts:
            handle.write(json.dumps(asdict(artifact), ensure_ascii=False) + "\n")


def write_summary(path: Path, model: str, artifacts: list[Artifact], records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["signal"]].append(record)

    by_signal: dict[str, Any] = {}
    for signal in SIGNALS:
        rows = grouped.get(signal, [])
        completed = [row for row in rows if row["status"] == "completed"]
        by_signal[signal] = {
            "planned_artifacts": sum(1 for artifact in artifacts if artifact.signal == signal),
            "attempted": len(rows),
            "completed_structured_outputs": len(completed),
            "malformed_or_api_error": sum(1 for row in rows if row["status"] != "completed"),
            "automatic_screen_pass": sum(
                1 for row in completed if row.get("validation", {}).get("automatic_screen_pass") is True
            ),
            "manual_review_required": signal in {"task_code", "educational_qa_mcq_general", "factual_restraint"},
            "total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in rows),
            "cost": sum(row.get("usage", {}).get("cost", 0.0) for row in rows),
        }

    summary = {
        "experiment": "deepseek_grounded_150_qualification",
        "model": model,
        "architecture": "grounded artifact -> one model-rendered final record -> local screen/manual review",
        "planned_total_artifacts": len(artifacts),
        "recorded_attempts": len(records),
        "structured_outputs_completed": sum(1 for record in records if record["status"] == "completed"),
        "malformed_or_api_error": sum(1 for record in records if record["status"] != "completed"),
        "automatic_screen_pass": sum(
            1 for record in records
            if record["status"] == "completed"
            and record.get("validation", {}).get("automatic_screen_pass") is True
        ),
        "total_tokens": sum(record.get("usage", {}).get("total_tokens", 0) for record in records),
        "total_cost": sum(record.get("usage", {}).get("cost", 0.0) for record in records),
        "by_signal": by_signal,
        "interpretation_note": (
            "automatic_screen_pass is an objective/structural screen only. "
            "task_code, educational_qa_mcq_general, and factual_restraint require manual review."
        ),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_review_files(output_dir: Path, artifacts: list[Artifact], records: list[dict[str, Any]]) -> None:
    artifact_map = {artifact.artifact_id: artifact for artifact in artifacts}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["signal"]].append(record)

    for signal in SIGNALS:
        lines = [f"# Manual Review: {signal}", ""]
        for record in grouped.get(signal, []):
            artifact = artifact_map[record["artifact_id"]]
            lines.extend([
                f"## {record['artifact_id']} — {record['family']}",
                "",
                f"Status: `{record['status']}`",
                "",
                "### Grounded artifact",
                "",
                "```json",
                json.dumps(artifact.payload, indent=2, ensure_ascii=False),
                "```",
                "",
            ])
            if record["status"] == "completed":
                lines.extend([
                    "### Generated output",
                    "",
                    "```json",
                    json.dumps(record["output"], indent=2, ensure_ascii=False),
                    "```",
                    "",
                    "### Automatic screen",
                    "",
                    "```json",
                    json.dumps(record["validation"], indent=2, ensure_ascii=False),
                    "```",
                    "",
                ])
            else:
                lines.extend([
                    "### Failure",
                    "",
                    "```json",
                    json.dumps({key: value for key, value in record.items() if key not in {"signal", "family", "artifact_id"}}, indent=2, ensure_ascii=False),
                    "```",
                    "",
                ])
        (output_dir / f"review_{signal}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-only, 150-artifact grounded qualification harness.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true", help="Write/validate artifact manifest without API calls.")
    parser.add_argument("--dry-run", action="store_true", help="Alias of --prepare-only.")
    parser.add_argument("--resume", action="store_true", help="Skip artifact IDs already recorded in output records.jsonl.")
    args = parser.parse_args()

    artifacts = build_all_artifacts()
    preflight_validate(artifacts)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"logs/{stamp}_deepseek_grounded_150_qualification")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "artifact_manifest.jsonl"
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    write_manifest(manifest_path, artifacts)

    counts = Counter(artifact.signal for artifact in artifacts)
    plan = {
        "model": args.model,
        "architecture": "grounded artifact -> one model-rendered final record -> local screen/manual review",
        "artifact_counts": dict(counts),
        "total_artifacts": len(artifacts),
        "paid_api_calls_without_retries": len(artifacts),
        "output_dir": str(output_dir),
        "preflight": "passed",
    }
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))

    if args.prepare_only or args.dry_run:
        print(f"\nPrepared manifest without API calls: {manifest_path}")
        return 0

    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing from .env or shell environment")

    if records_path.exists() and not args.resume:
        raise SystemExit(
            f"{records_path} already exists. Use --resume to continue without duplicate calls, "
            "or choose a new --output-dir."
        )

    completed_ids = read_completed_ids(records_path) if args.resume else set()
    pending = [artifact for artifact in artifacts if artifact.artifact_id not in completed_ids]
    print(f"Pending paid calls: {len(pending)}; already recorded: {len(completed_ids)}")

    with httpx.Client(timeout=180.0) as client:
        for index, artifact in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {artifact.signal} | {artifact.family} | {artifact.artifact_id}")
            result = call_openrouter(
                client,
                api_key=api_key,
                model=args.model,
                artifact=artifact,
                max_retries=args.max_retries,
            )
            record: dict[str, Any] = {
                "artifact_id": artifact.artifact_id,
                "signal": artifact.signal,
                "family": artifact.family,
                "model_requested": args.model,
                **result,
            }
            if result["status"] == "completed":
                record["validation"] = validate_output(artifact, result["output"])
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = read_records(records_path)
    summary = write_summary(summary_path, args.model, artifacts, records)
    write_review_files(output_dir, artifacts, records)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {records_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved per-signal review files under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
