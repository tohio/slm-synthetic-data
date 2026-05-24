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

    @staticmethod
    def _function_name(base: str, variant: int) -> str:
        """Keep the primary grounded name natural while retaining collision separation."""
        return base if variant == 0 else f"{base}_{variant}"

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
        n, mode_i, min_length_i, variant = self._decode(index, (len(self.NOUNS), 2, 16, 1000003))
        noun = self.NOUNS[n]
        plural = f"{noun}s"
        normalizer = "lower" if mode_i == 0 else "upper"
        min_length = min_length_i + 1
        fn = self._function_name(f"count_clean_{plural}_{normalizer}_{min_length}", variant)
        code = (
            f"def {fn}({plural}):\n"
            "    counts = {}\n"
            f"    for {noun} in {plural}:\n"
            f"        cleaned = {noun}.strip().{normalizer}()\n"
            f"        if len(cleaned) >= {min_length}:\n"
            "            counts[cleaned] = counts.get(cleaned, 0) + 1\n"
            "    return counts"
        )
        return {
            "code": code,
            "behavior_contract": f"Take a list of {noun} strings, strip and convert each complete string to {normalizer}case, retain normalized strings with length at least {min_length}, return their counts, and do not mutate the input.",
        }

    def _build_filter_sort_projection(self, index: int) -> dict[str, object]:
        out_i, metric_i, threshold_i, descending_i, variant = self._decode(index, (len(self.TITLES), len(self.METRICS), 71, 2, 1000003))
        output = self.TITLES[out_i]
        metric = self.METRICS[metric_i]
        threshold = 20 + threshold_i
        reverse = bool(descending_i)
        compare = ">=" if reverse else "<="
        direction = "descending" if reverse else "ascending"
        fn = self._function_name(f"select_{output}s_by_{metric}_{threshold}_{direction}", variant)
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
        noun_i, value_i, minimum_i, variant = self._decode(index, (len(self.NOUNS), len(self.VALUES), 101, 1000003))
        key = self.NOUNS[noun_i]
        value = self.VALUES[value_i]
        minimum = minimum_i
        fn = self._function_name(f"total_{value}_by_{key}_min_{minimum}", variant)
        code = (
            f"def {fn}(records):\n"
            "    totals = {}\n"
            "    for row in records:\n"
            f"        if row[\"{value}\"] >= {minimum}:\n"
            f"            group = row[\"{key}\"]\n"
            f"            totals[group] = totals.get(group, 0) + row[\"{value}\"]\n"
            "    return totals"
        )
        return {
            "code": code,
            "behavior_contract": f"Take a list of dictionaries with {key} and {value}, keep entries whose {value} is at least {minimum}, sum retained {value} by {key}, return a new dictionary, and do not mutate inputs.",
        }

    def _build_grouped_average_threshold(self, index: int) -> dict[str, object]:
        noun_i, value_i, threshold_i, variant = self._decode(index, (len(self.NOUNS), len(self.VALUES), 901, 1000003))
        key = self.NOUNS[noun_i]
        value = self.VALUES[value_i]
        threshold = 10 + threshold_i
        fn = self._function_name(f"qualifying_{value}_averages_by_{key}", variant)
        code = (
            f"def {fn}(records):\n"
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
        margin_i, variant = self._decode(index, (101, 1000003))
        margin = margin_i
        fn = self._function_name(f"compare_pairs_with_margin_{margin}", variant)
        code = (
            f"def {fn}(first, second):\n"
            "    counts = {\"first_ahead\": 0, \"second_ahead\": 0, \"within_margin\": 0}\n"
            "    for left, right in zip(first, second):\n"
            f"        if left - right > {margin}:\n"
            "            counts[\"first_ahead\"] += 1\n"
            f"        elif right - left > {margin}:\n"
            "            counts[\"second_ahead\"] += 1\n"
            "        else:\n"
            "            counts[\"within_margin\"] += 1\n"
            "    return counts"
        )
        return {
            "code": code,
            "behavior_contract": f"Take two equal-length integer lists, compare paired values using a margin of {margin}, count pairs where first is ahead by more than the margin, second is ahead by more than the margin, or values are within the margin, and do not mutate inputs.",
        }

    def _build_nested_transform(self, index: int) -> dict[str, object]:
        mode, factor, cutoff_i, variant = self._decode(index, (3, 19, 101, 100003))
        amount = factor + 2
        cutoff = cutoff_i
        if mode == 0:
            body = f"[[value * {amount} for value in row if value >= {cutoff}] for row in rows]"
            contract = f"retain integers at least {cutoff} and multiply each retained integer by {amount}"
        elif mode == 1:
            body = f"[[value for value in row if value > {cutoff}] for row in rows]"
            contract = f"keep only integers greater than {cutoff}"
        else:
            body = f"[[value + {amount} for value in row if value >= {cutoff}] for row in rows]"
            contract = f"retain integers at least {cutoff} and add {amount} to each retained integer"
        fn = self._function_name(f"transform_rows_{mode}_{amount}_{cutoff}", variant)
        code = f"def {fn}(rows):\n    return {body}"
        return {
            "code": code,
            "behavior_contract": f"Take a nested list of integers, {contract} within each row while preserving structure and order, return a new nested list, and do not mutate inputs.",
        }

    def _build_selection_by_total(self, index: int) -> dict[str, object]:
        noun_i, threshold_i, variant = self._decode(index, (len(self.NOUNS), 1801, 1009))
        key = self.NOUNS[noun_i]
        threshold = 50 + threshold_i
        fn = self._function_name(f"{key}s_over_total_{threshold}", variant)
        code = (
            f"def {fn}(records):\n"
            f"    return [row[\"{key}\"] for row in records if sum(row[\"values\"]) > {threshold}]"
        )
        return {
            "code": code,
            "behavior_contract": f"Take a list of dictionaries with {key} and a list of integer values, return {key} values whose sums exceed {threshold} in original order, and do not mutate inputs.",
        }

    def _build_dictionary_keywise_sum(self, index: int) -> dict[str, object]:
        noun_i, minimum_i, variant = self._decode(index, (len(self.NOUNS), 501, 1000003))
        noun = self.NOUNS[noun_i]
        minimum = minimum_i
        fn = self._function_name(f"combine_{noun}_counts_min_{minimum}", variant)
        code = (
            f"def {fn}(first, second):\n"
            "    result = {}\n"
            "    for key in set(first) | set(second):\n"
            "        total = first.get(key, 0) + second.get(key, 0)\n"
            f"        if total >= {minimum}:\n"
            "            result[key] = total\n"
            "    return result"
        )
        return {
            "code": code,
            "behavior_contract": f"Take two dictionaries of {noun} counts, sum values over the union of keys using zero for missing keys, retain totals at least {minimum}, return a new dictionary, and do not mutate inputs.",
        }
