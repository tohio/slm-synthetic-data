from __future__ import annotations

from slm_synth.artifacts.base import GroundedArtifact


class TaskCodeArtifactFactory:
    """Create valid short Python functions for instruction-reversal rendering."""

    FAMILIES = (
        "normalized_counting",
        "filter_sort_projection",
        "grouped_totals",
        "grouped_average_threshold",
        "paired_comparison_counts",
        "nested_transform",
        "selection_by_total",
        "dictionary_keywise_sum",
    )
    NOUNS = ("tag", "label", "status", "category", "topic", "region", "team", "group")
    TITLES = ("title", "name", "identifier", "label")
    METRICS = ("rating", "score", "priority", "quality", "rank", "weight")
    VALUES = ("hours", "units", "points", "cost", "sales", "items")

    @staticmethod
    def _decode(value: int, radices: tuple[int, ...]) -> list[int]:
        result: list[int] = []
        for radix in radices:
            result.append(value % radix)
            value //= radix
        return result

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        family = self.FAMILIES[index % len(self.FAMILIES)]
        payload = getattr(self, f"_build_{family}")(index // len(self.FAMILIES))
        return GroundedArtifact(
            signal="task_code",
            family=family,
            artifact_id=f"task_code_{family}_{index + 1:09d}",
            payload=payload,
        )

    def _build_normalized_counting(self, index: int) -> dict[str, object]:
        n, variant = self._decode(index, (len(self.NOUNS), 1000003))
        noun = self.NOUNS[n]
        plural = f"{noun}s"
        fn = f"count_clean_{plural}_{variant}"
        code = (
            f"def {fn}({plural}):\n"
            "    counts = {}\n"
            f"    for {noun} in {plural}:\n"
            f"        cleaned = {noun}.strip().lower()\n"
            "        if cleaned:\n"
            "            counts[cleaned] = counts.get(cleaned, 0) + 1\n"
            "    return counts"
        )
        return {
            "code": code,
            "behavior_contract": f"Take a list of {noun} strings, strip and lowercase each complete string, omit empty normalized strings, and return counts without mutating the input.",
        }

    def _build_filter_sort_projection(self, index: int) -> dict[str, object]:
        out_i, metric_i, threshold_i, descending_i, variant = self._decode(index, (len(self.TITLES), len(self.METRICS), 71, 2, 1000003))
        output = self.TITLES[out_i]
        metric = self.METRICS[metric_i]
        threshold = 20 + threshold_i
        reverse = bool(descending_i)
        compare = ">=" if reverse else "<="
        direction = "descending" if reverse else "ascending"
        fn = f"select_{output}s_by_{metric}_{threshold}_{direction}_{variant}"
        code = (
            f"def {fn}(records):\n"
            f"    kept = [row for row in records if row[\"{metric}\"] {compare} {threshold}]\n"
            f"    kept = sorted(kept, key=lambda row: row[\"{metric}\"], reverse={str(reverse)})\n"
            f"    return [row[\"{output}\"] for row in kept]"
        )
        return {
            "code": code,
            "behavior_contract": f"Take a list of dictionaries, keep records with {metric} {compare} {threshold}, sort by {metric} in {direction} order, return their {output} values, and do not mutate inputs.",
        }

    def _build_grouped_totals(self, index: int) -> dict[str, object]:
        noun_i, value_i, variant = self._decode(index, (len(self.NOUNS), len(self.VALUES), 1000003))
        key = self.NOUNS[noun_i]
        value = self.VALUES[value_i]
        code = (
            f"def total_{value}_by_{key}_{variant}(records):\n"
            "    totals = {}\n"
            "    for row in records:\n"
            f"        group = row[\"{key}\"]\n"
            f"        totals[group] = totals.get(group, 0) + row[\"{value}\"]\n"
            "    return totals"
        )
        return {
            "code": code,
            "behavior_contract": f"Take a list of dictionaries with {key} and {value}, sum {value} by {key}, return a new dictionary, and do not mutate inputs.",
        }

    def _build_grouped_average_threshold(self, index: int) -> dict[str, object]:
        noun_i, value_i, threshold_i, variant = self._decode(index, (len(self.NOUNS), len(self.VALUES), 901, 1000003))
        key = self.NOUNS[noun_i]
        value = self.VALUES[value_i]
        threshold = 10 + threshold_i
        code = (
            f"def qualifying_{value}_averages_by_{key}_{variant}(records):\n"
            "    totals = {}\n"
            "    counts = {}\n"
            "    for row in records:\n"
            f"        group = row[\"{key}\"]\n"
            f"        totals[group] = totals.get(group, 0) + row[\"{value}\"]\n"
            "        counts[group] = counts.get(group, 0) + 1\n"
            f"    return {{g: totals[g] / counts[g] for g in totals if totals[g] / counts[g] >= {threshold}}}"
        )
        return {
            "code": code,
            "behavior_contract": f"Compute average {value} per {key} from a list of dictionaries, retain averages at least {threshold}, return a dictionary, and do not mutate inputs.",
        }

    def _build_paired_comparison_counts(self, index: int) -> dict[str, object]:
        variant = index % 1000003
        code = (
            f"def compare_pairs_{variant}(first, second):\n"
            "    counts = {\"first_higher\": 0, \"second_higher\": 0, \"equal\": 0}\n"
            "    for left, right in zip(first, second):\n"
            "        if left > right:\n"
            "            counts[\"first_higher\"] += 1\n"
            "        elif right > left:\n"
            "            counts[\"second_higher\"] += 1\n"
            "        else:\n"
            "            counts[\"equal\"] += 1\n"
            "    return counts"
        )
        return {
            "code": code,
            "behavior_contract": "Take two equal-length lists of integers, compare paired values, return counts for first_higher, second_higher, and equal, and do not mutate inputs.",
        }

    def _build_nested_transform(self, index: int) -> dict[str, object]:
        mode, factor, variant = self._decode(index, (3, 8, 100003))
        multiplier = factor + 2
        if mode == 0:
            body = f"[[value * {multiplier} for value in row] for row in rows]"
            contract = f"multiply each integer by {multiplier}"
        elif mode == 1:
            body = "[[value for value in row if value > 0] for row in rows]"
            contract = "keep only positive integers"
        else:
            body = f"[[value + {multiplier} for value in row] for row in rows]"
            contract = f"add {multiplier} to each integer"
        code = f"def transform_rows_{variant}_{mode}_{multiplier}(rows):\n    return {body}"
        return {
            "code": code,
            "behavior_contract": f"Take a nested list of integers, {contract} within each row while preserving structure and order, return a new nested list, and do not mutate inputs.",
        }

    def _build_selection_by_total(self, index: int) -> dict[str, object]:
        noun_i, threshold_i, variant = self._decode(index, (len(self.NOUNS), 1801, 1009))
        key = self.NOUNS[noun_i]
        threshold = 50 + threshold_i
        code = (
            f"def {key}s_over_total_{threshold}_{variant}(records):\n"
            f"    return [row[\"{key}\"] for row in records if sum(row[\"values\"]) > {threshold}]"
        )
        return {
            "code": code,
            "behavior_contract": f"Take a list of dictionaries with {key} and a list of integer values, return {key} values whose sums exceed {threshold} in original order, and do not mutate inputs.",
        }

    def _build_dictionary_keywise_sum(self, index: int) -> dict[str, object]:
        noun_i, variant = self._decode(index, (len(self.NOUNS), 1000003))
        noun = self.NOUNS[noun_i]
        code = (
            f"def combine_{noun}_counts_{variant}(first, second):\n"
            "    result = {}\n"
            "    for key in set(first) | set(second):\n"
            "        result[key] = first.get(key, 0) + second.get(key, 0)\n"
            "    return result"
        )
        return {
            "code": code,
            "behavior_contract": f"Take two dictionaries of {noun} counts, sum values over the union of keys using zero for missing keys, return a new dictionary, and do not mutate inputs.",
        }
