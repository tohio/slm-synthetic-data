from __future__ import annotations

from slm_synth.artifacts.base import GroundedArtifact


class EducationalQAMCQMathArtifactFactory:
    FAMILIES = ("integer_expression", "missing_operand", "exact_division", "two_step_quantity", "unique_numeric_comparison")
    SETTINGS = ("folders", "markers", "tickets", "cards", "samples", "labels", "notebooks", "vouchers")

    @staticmethod
    def _decode(value: int, radices: tuple[int, ...]) -> list[int]:
        result: list[int] = []
        for radix in radices:
            result.append(value % radix)
            value //= radix
        return result

    @staticmethod
    def _choices(answer: int, index: int) -> list[str]:
        offsets = ([-7, 4, 11], [-5, 6, 13], [-9, 3, 8], [-4, 7, 15])[index % 4]
        values = [str(answer + offset) for offset in offsets]
        values.insert(index % 4, str(answer))
        return values

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        family = self.FAMILIES[index % len(self.FAMILIES)]
        payload = getattr(self, f"_build_{family}")(index // len(self.FAMILIES))
        return GroundedArtifact("educational_qa_mcq_math", family, f"educational_qa_mcq_math_{family}_{index + 1:09d}", payload)

    def _base(self, question_instruction: str, expression: str, numbers: list[str], answer: int, index: int) -> dict[str, object]:
        return {
            "question_instruction": question_instruction,
            "required_numeric_literals": numbers,
            "choices": self._choices(answer, index),
            "answer": str(answer),
            "correct_index": index % 4,
            "expression": expression,
        }

    def _build_integer_expression(self, index: int) -> dict[str, object]:
        a, b, c, d = self._decode(index, (151, 71, 8, 91))
        a, b, c, d = a + 20, b + 3, c + 2, d + 1
        expression = f"({a} + {b}) * {c} - {d}"
        answer = (a + b) * c - d
        return self._base(f"Ask the learner to evaluate {expression}.", expression, [str(a), str(b), str(c), str(d)], answer, index)

    def _build_missing_operand(self, index: int) -> dict[str, object]:
        multiplier_i, unknown_i, offset_i = self._decode(index, (41, 2000, 1001))
        multiplier, unknown, offset = multiplier_i + 2, unknown_i + 2, offset_i + 1
        total = multiplier * unknown + offset
        equation = f"{multiplier} * ? + {offset} = {total}"
        expression = f"({total} - {offset}) / {multiplier}"
        return self._base(f"Ask which integer replaces the question mark in {equation}.", expression, [str(multiplier), str(offset), str(total)], unknown, index)

    def _build_exact_division(self, index: int) -> dict[str, object]:
        setting_i, groups_i, per_i = self._decode(index, (len(self.SETTINGS), 1001, 1001))
        groups, per = groups_i + 10, per_i + 5
        total = groups * per
        setting = self.SETTINGS[setting_i]
        return self._base(
            f"Create a question where {total} {setting} are divided equally among {groups} containers and ask how many are in each container.",
            f"{total} / {groups}", [str(total), str(groups)], per, index,
        )

    def _build_two_step_quantity(self, index: int) -> dict[str, object]:
        setting_i, first_i, second_i, remain_i = self._decode(index, (len(self.SETTINGS), 151, 131, 251))
        first, second, remain = first_i + 20, second_i + 10, remain_i + 20
        start = first + second + remain
        setting = self.SETTINGS[setting_i]
        return self._base(
            f"Create a remaining-quantity question about {setting}: begin with {start}, remove {first}, then remove {second} more.",
            f"{start} - {first} - {second}", [str(start), str(first), str(second)], remain, index,
        )

    def _build_unique_numeric_comparison(self, index: int) -> dict[str, object]:
        a, b, c = self._decode(index, (101, 91, 81))
        x, y, z = a + 11, b + 20, c + 30
        expressions = [f"{x} * 3", f"{y} + 47", f"{z} + 80"]
        values = [x * 3, y + 47, z + 80]
        highest = max(values)
        if values.count(highest) != 1:
            z += 101
            expressions[2] = f"{z} + 80"
            values[2] = z + 80
            highest = max(values)
        winner = expressions[values.index(highest)]
        numbers = [token for expression in expressions for token in expression.replace("*", " ").replace("+", " ").split()]
        return self._base(
            "Ask for the largest numeric value among: " + ", ".join(expressions) + ".",
            winner, numbers, highest, index,
        )
