from __future__ import annotations

from slm_synth.artifacts.base import GroundedArtifact


class ArithmeticArtifactFactory:
    """Create distinct, locally-verifiable integer-arithmetic backbones.

    Each family decodes its family-local sequence number into a large finite
    parameter space. This keeps operands practical while supporting hundreds of
    thousands of distinct artifacts per family without paraphrase reuse.
    """

    FAMILIES = (
        "direct_expression",
        "missing_operand",
        "two_step_remaining_quantity",
        "exact_allocation",
        "unique_numeric_comparison",
    )
    SETTINGS = {
        "missing_operand": ("storage labels", "sample tubes", "paint jars", "repair bolts"),
        "two_step_remaining_quantity": ("tickets", "badges", "vouchers", "notebooks"),
        "exact_allocation": ("trays", "cartons", "bins", "racks"),
    }

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
        family_index = index // len(self.FAMILIES)
        method = getattr(self, f"_build_{family}")
        payload = method(family_index)
        return GroundedArtifact(
            signal="arithmetic",
            family=family,
            artifact_id=f"arithmetic_{family}_{index + 1:09d}",
            payload=payload,
        )

    def _build_direct_expression(self, index: int) -> dict[str, object]:
        x, y, z, w = self._decode(index, (151, 73, 8, 91))
        a, b, c, d = 40 + x, 7 + y, 2 + z, 3 + w
        expression = f"({a} - {b}) * {c} + {d}"
        return {
            "instruction": "Create a direct integer-expression question.",
            "expression": expression,
            "required_numeric_literals": [str(a), str(b), str(c), str(d)],
            "answer": str((a - b) * c + d),
        }

    def _build_missing_operand(self, index: int) -> dict[str, object]:
        original_offset, added_offset, setting_id = self._decode(index, (700, 400, 4))
        original, added = 25 + original_offset, 11 + added_offset
        total = original + added
        setting = self.SETTINGS["missing_operand"][setting_id]
        return {
            "instruction": "Create a word problem asking for the unknown starting quantity.",
            "setting": setting,
            "facts": [f"{added} were added", f"the final total is {total}"],
            "expression": f"{total} - {added}",
            "required_numeric_literals": [str(added), str(total)],
            "answer": str(original),
        }

    def _build_two_step_remaining_quantity(self, index: int) -> dict[str, object]:
        first_offset, second_offset, remaining_offset, setting_id = self._decode(index, (300, 250, 400, 4))
        first, second, remaining = 30 + first_offset, 20 + second_offset, 50 + remaining_offset
        start = first + second + remaining
        setting = self.SETTINGS["two_step_remaining_quantity"][setting_id]
        return {
            "instruction": "Create a two-step remaining-quantity word problem.",
            "setting": setting,
            "facts": [f"start with {start}", f"remove {first}", f"remove {second} more"],
            "expression": f"{start} - {first} - {second}",
            "required_numeric_literals": [str(start), str(first), str(second)],
            "answer": str(remaining),
        }

    def _build_exact_allocation(self, index: int) -> dict[str, object]:
        per_offset, container_offset, setting_id = self._decode(index, (500, 1000, 4))
        per_container, containers = 12 + per_offset, 18 + container_offset
        total = per_container * containers
        setting = self.SETTINGS["exact_allocation"][setting_id]
        return {
            "instruction": "Create an exact-allocation question asking for the number of containers required.",
            "setting": setting,
            "facts": [f"{total} items total", f"{per_container} items per container"],
            "expression": f"{total} / {per_container}",
            "required_numeric_literals": [str(total), str(per_container)],
            "answer": str(containers),
        }

    def _build_unique_numeric_comparison(self, index: int) -> dict[str, object]:
        base = 80 + index
        expressions = [f"{base - 9} + 4", f"{base - 18} + 12", f"{base} + 7"]
        return {
            "instruction": "Ask for the unique largest numeric value among the three supplied expressions.",
            "expressions": expressions,
            "winning_expression": expressions[2],
            "expression": expressions[2],
            "required_numeric_literals": [str(base - 9), "4", str(base - 18), "12", str(base), "7"],
            "answer": str(base + 7),
        }
