from __future__ import annotations

from slm_synth.pretrain.artifacts.base import GroundedArtifact
from slm_synth.pretrain.artifacts.task_code_advanced import ADVANCED_TASK_CODE_SPECS
from slm_synth.pretrain.artifacts.task_code_advanced_2 import ADVANCED_TASK_CODE_SPECS_2
from slm_synth.pretrain.artifacts.task_code_advanced_3 import ADVANCED_TASK_CODE_SPECS_3
from slm_synth.pretrain.artifacts.task_code_advanced_4 import ADVANCED_TASK_CODE_SPECS_4


class TaskCodeArtifactFactory:
    """Create a finite catalog of distinct, valid Python algorithm records."""

    CORE_SPECS = (
        (
            "normalized_string_counts",
            "Write a Python function that strips and lowercases each string, ignores empty results, returns occurrence counts, and does not mutate the input.",
            "def count_normalized_strings(values):\n    counts = {}\n    for value in values:\n        cleaned = value.strip().lower()\n        if cleaned:\n            counts[cleaned] = counts.get(cleaned, 0) + 1\n    return counts",
        ),
        (
            "filter_sort_projection",
            "Write a Python function that keeps records meeting a minimum score, sorts the retained records by descending score, returns their labels, and does not mutate the input.",
            "def select_ranked_labels(records, minimum):\n    kept = [row for row in records if row[\"score\"] >= minimum]\n    ordered = sorted(kept, key=lambda row: row[\"score\"], reverse=True)\n    return [row[\"label\"] for row in ordered]",
        ),
        (
            "grouped_numeric_totals",
            "Write a Python function that sums each record's amount by category, returns a new category-to-total dictionary, and does not mutate the records.",
            "def totals_by_category(records):\n    totals = {}\n    for row in records:\n        category = row[\"category\"]\n        totals[category] = totals.get(category, 0) + row[\"amount\"]\n    return totals",
        ),
        (
            "grouped_numeric_averages",
            "Write a Python function that computes the arithmetic mean of record values for each group, returns a new dictionary, and does not mutate the records.",
            "def averages_by_group(records):\n    totals = {}\n    counts = {}\n    for row in records:\n        group = row[\"group\"]\n        totals[group] = totals.get(group, 0) + row[\"value\"]\n        counts[group] = counts.get(group, 0) + 1\n    return {group: totals[group] / counts[group] for group in totals}",
        ),
        (
            "paired_margin_classification",
            "Write a Python function that compares paired numbers from two equal-length sequences, classifies each pair as left ahead, right ahead, or within a supplied margin, returns the three counts, and does not mutate either sequence.",
            "def classify_pair_margins(left_values, right_values, margin):\n    counts = {\"left_ahead\": 0, \"right_ahead\": 0, \"within_margin\": 0}\n    for left, right in zip(left_values, right_values):\n        if left - right > margin:\n            counts[\"left_ahead\"] += 1\n        elif right - left > margin:\n            counts[\"right_ahead\"] += 1\n        else:\n            counts[\"within_margin\"] += 1\n    return counts",
        ),
        (
            "nested_filter_transform",
            "Write a Python function that processes every row in a nested integer list, keeps nonnegative values, doubles each retained value, preserves row boundaries and order, and does not mutate the input.",
            "def double_nonnegative_rows(rows):\n    return [[value * 2 for value in row if value >= 0] for row in rows]",
        ),
        (
            "select_by_nested_total",
            "Write a Python function that returns the names of records whose list of values has a sum above a supplied threshold, preserves input order, and does not mutate the records.",
            "def names_above_value_total(records, threshold):\n    return [row[\"name\"] for row in records if sum(row[\"values\"]) > threshold]",
        ),
        (
            "dictionary_keywise_sum",
            "Write a Python function that adds values from two dictionaries over their union of keys, treats missing values as zero, returns a new dictionary, and does not mutate either input.",
            "def add_dictionary_values(first, second):\n    result = {}\n    for key in set(first) | set(second):\n        result[key] = first.get(key, 0) + second.get(key, 0)\n    return result",
        ),
        (
            "stable_unique_values",
            "Write a Python function that removes duplicate hashable values while preserving first-occurrence order, returns a new list, and does not mutate the input.",
            "def unique_in_order(values):\n    seen = set()\n    result = []\n    for value in values:\n        if value not in seen:\n            seen.add(value)\n            result.append(value)\n    return result",
        ),
        (
            "running_totals",
            "Write a Python function that returns the cumulative sum after each number in a sequence and does not mutate the input.",
            "def cumulative_sums(values):\n    result = []\n    total = 0\n    for value in values:\n        total += value\n        result.append(total)\n    return result",
        ),
        (
            "fixed_size_chunks",
            "Write a Python function that divides a sequence into consecutive lists of a positive maximum size, includes a shorter final chunk when needed, and does not mutate the input.",
            "def chunk_values(values, size):\n    if size <= 0:\n        raise ValueError(\"size must be positive\")\n    return [list(values[index:index + size]) for index in range(0, len(values), size)]",
        ),
        (
            "group_records_by_key",
            "Write a Python function that groups records by their category field, preserves record order within each group, returns copied records in new lists, and does not mutate the input.",
            "def group_records_by_category(records):\n    grouped = {}\n    for row in records:\n        grouped.setdefault(row[\"category\"], []).append(row.copy())\n    return grouped",
        ),
        (
            "merge_sorted_sequences",
            "Write a Python function that merges two ascending numeric sequences into one ascending list in linear time and does not mutate either input.",
            "def merge_sorted_values(first, second):\n    merged = []\n    left = right = 0\n    while left < len(first) and right < len(second):\n        if first[left] <= second[right]:\n            merged.append(first[left])\n            left += 1\n        else:\n            merged.append(second[right])\n            right += 1\n    merged.extend(first[left:])\n    merged.extend(second[right:])\n    return merged",
        ),
        (
            "sliding_window_sums",
            "Write a Python function that returns the sum of every consecutive window of a positive size, returns an empty list when the window exceeds the sequence, and does not mutate the input.",
            "def sliding_sums(values, width):\n    if width <= 0:\n        raise ValueError(\"width must be positive\")\n    if width > len(values):\n        return []\n    total = sum(values[:width])\n    result = [total]\n    for index in range(width, len(values)):\n        total += values[index] - values[index - width]\n        result.append(total)\n    return result",
        ),
        (
            "maximum_record_per_group",
            "Write a Python function that selects the highest-scoring record in each group, keeps the first record on ties, returns copied records in a new dictionary, and does not mutate the input.",
            "def best_record_by_group(records):\n    best = {}\n    for row in records:\n        group = row[\"group\"]\n        if group not in best or row[\"score\"] > best[group][\"score\"]:\n            best[group] = row.copy()\n    return best",
        ),
        (
            "matrix_transpose",
            "Write a Python function that transposes a rectangular nested list, rejects rows of unequal length, returns new lists, and does not mutate the matrix.",
            "def transpose_rectangular(matrix):\n    if not matrix:\n        return []\n    width = len(matrix[0])\n    if any(len(row) != width for row in matrix):\n        raise ValueError(\"matrix must be rectangular\")\n    return [[row[column] for row in matrix] for column in range(width)]",
        ),
        (
            "predicate_partition",
            "Write a Python function that partitions values into matching and nonmatching lists using a supplied predicate, preserves order, returns both lists, and does not mutate the input.",
            "def partition_values(values, predicate):\n    matching = []\n    remaining = []\n    for value in values:\n        if predicate(value):\n            matching.append(value)\n        else:\n            remaining.append(value)\n    return matching, remaining",
        ),
        (
            "unique_record_index",
            "Write a Python function that indexes copied records by their id field, raises an error for a repeated id, returns a new dictionary, and does not mutate the records.",
            "def index_records_by_id(records):\n    indexed = {}\n    for row in records:\n        identifier = row[\"id\"]\n        if identifier in indexed:\n            raise ValueError(\"duplicate id\")\n        indexed[identifier] = row.copy()\n    return indexed",
        ),
        (
            "grouped_numeric_ranges",
            "Write a Python function that returns the minimum and maximum value observed for each group and does not mutate the records.",
            "def value_ranges_by_group(records):\n    ranges = {}\n    for row in records:\n        group = row[\"group\"]\n        value = row[\"value\"]\n        if group not in ranges:\n            ranges[group] = [value, value]\n        else:\n            ranges[group][0] = min(ranges[group][0], value)\n            ranges[group][1] = max(ranges[group][1], value)\n    return {group: tuple(bounds) for group, bounds in ranges.items()}",
        ),
        (
            "balance_event_series",
            "Write a Python function that applies signed transaction amounts to an opening balance, returns the balance after every transaction, and does not mutate the transactions.",
            "def balances_after_transactions(opening, transactions):\n    balances = []\n    current = opening\n    for transaction in transactions:\n        current += transaction[\"amount\"]\n        balances.append(current)\n    return balances",
        ),
        (
            "stable_frequency_mode",
            "Write a Python function that returns the most frequent hashable value, resolves ties by first occurrence, returns None for empty input, and does not mutate the sequence.",
            "def first_mode(values):\n    if not values:\n        return None\n    counts = {}\n    first_positions = {}\n    for index, value in enumerate(values):\n        counts[value] = counts.get(value, 0) + 1\n        first_positions.setdefault(value, index)\n    return min(counts, key=lambda value: (-counts[value], first_positions[value]))",
        ),
        (
            "adjacent_differences",
            "Write a Python function that returns each value minus its immediate predecessor, returns an empty list for fewer than two values, and does not mutate the input.",
            "def adjacent_differences(values):\n    return [values[index] - values[index - 1] for index in range(1, len(values))]",
        ),
        (
            "nested_mapping_lookup",
            "Write a Python function that follows a sequence of keys through nested dictionaries, returns a supplied default when a key is absent or an intermediate value is not a dictionary, and does not mutate the mapping.",
            "def nested_lookup(mapping, keys, default=None):\n    current = mapping\n    for key in keys:\n        if not isinstance(current, dict) or key not in current:\n            return default\n        current = current[key]\n    return current",
        ),
        (
            "sparse_vector_dot_product",
            "Write a Python function that computes the dot product of two sparse vectors represented as dictionaries and does not mutate either dictionary.",
            "def sparse_dot_product(first, second):\n    if len(first) > len(second):\n        first, second = second, first\n    return sum(value * second.get(key, 0) for key, value in first.items())",
        ),
    )

    SPECS = CORE_SPECS + ADVANCED_TASK_CODE_SPECS + ADVANCED_TASK_CODE_SPECS_2 + ADVANCED_TASK_CODE_SPECS_3 + ADVANCED_TASK_CODE_SPECS_4
    FAMILIES = tuple(spec[0] for spec in SPECS)
    UNIQUE_CANDIDATE_CAPACITY = len(SPECS)

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        if not 0 <= index < self.UNIQUE_CANDIDATE_CAPACITY:
            raise ValueError(
                f"task_code index {index} exceeds unique candidate capacity "
                f"{self.UNIQUE_CANDIDATE_CAPACITY}"
            )
        family, task, code = self.SPECS[index]
        return GroundedArtifact(
            signal="task_code",
            family=family,
            artifact_id=f"task_code_{family}_{index + 1:09d}",
            payload={"task": task, "code": code},
        )
