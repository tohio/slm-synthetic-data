from __future__ import annotations

import ast
from fractions import Fraction

from slm_synth.pretrain.artifacts.base import GroundedArtifact


class EducationalQAMCQMathArtifactFactory:
    """Create finite, verified math MCQs across distinct relationships and contexts."""

    FAMILIES = (
        "expression_precedence", "missing_factor", "equal_sharing", "two_step_remaining",
        "expression_comparison", "fraction_of_quantity", "proportional_scaling", "rectangle_area",
        "rectangle_perimeter", "arithmetic_mean", "total_from_mean", "data_range",
        "combined_duration", "metric_conversion", "percentage_of_quantity", "percentage_discount",
        "percentage_increase", "two_rate_total", "linear_equation", "groups_plus_remainder",
        "arithmetic_sequence_sum", "complement_count", "map_scale", "combined_rectangle_area",
    )
    CONTEXTS = (
        ("an archive inventory review", "document folders"),
        ("a community garden supply plan", "seed packets"),
        ("a repair workshop service log", "replacement gears"),
        ("a science laboratory worksheet", "sample vials"),
        ("a school library circulation report", "returned books"),
        ("a food pantry distribution record", "meal kits"),
        ("a theater booking summary", "admission tickets"),
        ("a shipping depot manifest", "sealed parcels"),
        ("a wildlife clinic treatment log", "supply packs"),
        ("a museum catalog audit", "display labels"),
        ("a transit office operations log", "route notices"),
    )
    VARIANTS_PER_FAMILY = len(CONTEXTS)
    UNIQUE_CANDIDATE_CAPACITY = len(FAMILIES) * VARIANTS_PER_FAMILY

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        if not 0 <= index < self.UNIQUE_CANDIDATE_CAPACITY:
            raise ValueError(
                f"educational_qa_mcq_math index {index} exceeds unique candidate capacity "
                f"{self.UNIQUE_CANDIDATE_CAPACITY}"
            )
        family_index = index % len(self.FAMILIES)
        variant = index // len(self.FAMILIES)
        family = self.FAMILIES[family_index]
        question, expression, numeric_literals, answer = getattr(self, f"_build_{family}")(variant)
        choices = self._choices(answer, index)
        return GroundedArtifact(
            signal="educational_qa_mcq_math",
            family=family,
            artifact_id=f"educational_qa_mcq_math_{family}_{index + 1:09d}",
            payload={
                "question": question,
                "required_numeric_literals": [str(value) for value in numeric_literals],
                "choices": choices,
                "answer": str(answer),
                "correct_index": choices.index(str(answer)),
                "expression": expression,
            },
        )

    @staticmethod
    def _choices(answer: int, index: int) -> list[str]:
        magnitudes = (max(2, abs(answer) // 10), max(3, abs(answer) // 7), max(4, abs(answer) // 5))
        candidates = (answer - magnitudes[0], answer + magnitudes[0], answer + magnitudes[1], answer - magnitudes[2], answer + 1)
        distractors: list[str] = []
        for value in candidates:
            text = str(value)
            if value >= 0 and value != answer and text not in distractors:
                distractors.append(text)
            if len(distractors) == 3:
                break
        while len(distractors) < 3:
            candidate = answer + len(distractors) + 2
            if candidate != answer and str(candidate) not in distractors:
                distractors.append(str(candidate))
        distractors.insert(index % 4, str(answer))
        return distractors

    def _context(self, variant: int) -> tuple[str, str]:
        return self.CONTEXTS[variant]

    @staticmethod
    def _integer(expression: str) -> int:
        tree = ast.parse(expression, mode="eval")

        def evaluate(node: ast.AST) -> Fraction:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                return Fraction(node.value)
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div) and right:
                    return left / right
            raise ValueError("unsupported expression")

        fraction = evaluate(tree)
        if fraction.denominator != 1:
            raise ValueError("math artifact expression must have an integer result")
        return fraction.numerator

    def _build_expression_precedence(self, variant: int):
        context, item = self._context(variant)
        a, b, c, d = 16 + variant * 2, 5 + variant, 3 + variant % 4, 4 + variant % 5
        expression = f"({a} + {b}) * {c} - {d}"
        return f"In {context}, a calculation for {item} is written as ({a} + {b}) × {c} − {d}. What is its value?", expression, (a, b, c, d), self._integer(expression)

    def _build_missing_factor(self, variant: int):
        context, item = self._context(variant)
        factor, unknown, extra = 4 + variant % 6, 9 + variant * 2, 3 + variant
        total = factor * unknown + extra
        expression = f"({total} - {extra}) / {factor}"
        return f"According to {context}, {factor} equal groups of {item} plus {extra} loose items total {total}. How many are in each equal group?", expression, (factor, extra, total), unknown

    def _build_equal_sharing(self, variant: int):
        context, item = self._context(variant)
        groups, share = 4 + variant, 11 + variant * 2
        total = groups * share
        expression = f"{total} / {groups}"
        return f"During {context}, {total} {item} are divided equally among {groups} stations. How many does each station receive?", expression, (total, groups), share

    def _build_two_step_remaining(self, variant: int):
        context, item = self._context(variant)
        first, second, remaining = 12 + variant, 7 + variant * 2, 41 + variant * 3
        start = first + second + remaining
        expression = f"{start} - {first} - {second}"
        return f"A record from {context} begins with {start} {item}, sends out {first}, and later sends out {second}. How many remain?", expression, (start, first, second), remaining

    def _build_expression_comparison(self, variant: int):
        context, item = self._context(variant)
        a, b, c, d, e, f = 11 + variant, 4 + variant % 5, 37 + variant * 2, 19 + variant, 82 + variant * 3, 9 + variant
        values = (a * b, c + d, e - f)
        answer = max(values)
        expression = f"{a} * {b}" if answer == values[0] else f"{c} + {d}" if answer == values[1] else f"{e} - {f}"
        return f"For {context}, three proposed totals for {item} are {a} × {b}, {c} + {d}, and {e} − {f}. What is the largest resulting value?", expression, (a, b, c, d, e, f), answer

    def _build_fraction_of_quantity(self, variant: int):
        context, item = self._context(variant)
        denominator = 4 + variant % 5
        numerator = 2 + variant % (denominator - 1)
        unit = 12 + variant * 3
        total, answer = denominator * unit, numerator * unit
        expression = f"{total} * {numerator} / {denominator}"
        return f"In {context}, {numerator}/{denominator} of {total} {item} pass inspection. How many pass inspection?", expression, (numerator, denominator, total), answer

    def _build_proportional_scaling(self, variant: int):
        context, item = self._context(variant)
        base_units, per_unit, target_units = 3 + variant % 4, 6 + variant, 8 + variant
        base_total, answer = base_units * per_unit, target_units * per_unit
        expression = f"{base_total} / {base_units} * {target_units}"
        return f"A plan in {context} uses {base_total} {item} for {base_units} identical sections. At the same rate, how many are needed for {target_units} sections?", expression, (base_total, base_units, target_units), answer

    def _build_rectangle_area(self, variant: int):
        context, item = self._context(variant)
        length, width = 8 + variant, 5 + variant % 6
        expression = f"{length} * {width}"
        return f"A rectangular storage area in {context} for {item} measures {length} units by {width} units. What is its area?", expression, (length, width), length * width

    def _build_rectangle_perimeter(self, variant: int):
        context, item = self._context(variant)
        length, width = 12 + variant, 6 + variant % 5
        expression = f"2 * ({length} + {width})"
        return f"A rectangular boundary used in {context} around {item} is {length} units long and {width} units wide. What is the perimeter?", expression, (length, width), 2 * (length + width)

    def _build_arithmetic_mean(self, variant: int):
        context, item = self._context(variant)
        start, step = 12 + variant * 2, 4 + 2 * (variant % 4)
        values = (start, start + step, start + 2 * step, start + 3 * step)
        expression = f"({values[0]} + {values[1]} + {values[2]} + {values[3]}) / 4"
        return f"Four counts of {item} in {context} are {values[0]}, {values[1]}, {values[2]}, and {values[3]}. What is their arithmetic mean?", expression, values, sum(values) // 4

    def _build_total_from_mean(self, variant: int):
        context, item = self._context(variant)
        count, mean = 5 + variant, 13 + variant * 2
        expression = f"{count} * {mean}"
        return f"A summary from {context} reports {count} batches of {item} with a mean count of {mean}. What is the combined count?", expression, (count, mean), count * mean

    def _build_data_range(self, variant: int):
        context, item = self._context(variant)
        low = 8 + variant
        values = (low, low + 9 + variant, low + 4, low + 15 + variant * 2)
        expression = f"{max(values)} - {min(values)}"
        return f"Recorded {item} counts in {context} are {values[0]}, {values[1]}, {values[2]}, and {values[3]}. What is the range?", expression, values, max(values) - min(values)

    def _build_combined_duration(self, variant: int):
        context, item = self._context(variant)
        first, second = 24 + variant * 2, 31 + variant * 3
        expression = f"{first} + {second}"
        return f"In {context}, preparing {item} takes {first} minutes and completing the review takes {second} minutes. How many minutes are required altogether?", expression, (first, second), first + second

    def _build_metric_conversion(self, variant: int):
        context, item = self._context(variant)
        meters = 3 + variant
        expression = f"{meters} * 100"
        return f"A measured strip used for {item} in {context} is {meters} meters long. Using 1 meter = 100 centimeters, how many centimeters long is it?", expression, (meters, 1, 100), meters * 100

    def _build_percentage_of_quantity(self, variant: int):
        context, item = self._context(variant)
        percent = (10, 20, 25, 30, 40, 50, 60, 75, 80, 90, 15)[variant]
        total = 200 + variant * 40
        answer = total * percent // 100
        expression = f"{total} * {percent} / 100"
        return f"A report from {context} marks {percent}% of {total} {item} as complete. How many are complete?", expression, (percent, total), answer

    def _build_percentage_discount(self, variant: int):
        context, item = self._context(variant)
        percent, price = 5 * (variant + 1), 100 + variant * 20
        answer = price - price * percent // 100
        expression = f"{price} - {price} * {percent} / 100"
        return f"A purchase in {context} lists {item} at {price} dollars before a {percent}% discount. What is the discounted price?", expression, (price, percent), answer

    def _build_percentage_increase(self, variant: int):
        context, item = self._context(variant)
        percent = (10, 20, 25, 40, 50, 60, 75, 80, 100, 125, 15)[variant]
        original = 40 * (variant + 2)
        answer = original + original * percent // 100
        expression = f"{original} + {original} * {percent} / 100"
        return f"The planned quantity of {item} in {context} rises from {original} by {percent}%. What is the new quantity?", expression, (original, percent), answer

    def _build_two_rate_total(self, variant: int):
        context, item = self._context(variant)
        first_rate, first_time = 8 + variant, 3 + variant % 4
        second_rate, second_time = 13 + variant, 2 + variant % 3
        expression = f"{first_rate} * {first_time} + {second_rate} * {second_time}"
        answer = first_rate * first_time + second_rate * second_time
        return f"During {context}, {item} are processed at {first_rate} per hour for {first_time} hours and then {second_rate} per hour for {second_time} hours. How many are processed?", expression, (first_rate, first_time, second_rate, second_time), answer

    def _build_linear_equation(self, variant: int):
        context, item = self._context(variant)
        coefficient, solution, offset = 3 + variant, 7 + variant * 2, 4 + variant
        total = coefficient * solution - offset
        expression = f"({total} + {offset}) / {coefficient}"
        return f"A calibration equation in {context} for {item} is {coefficient}x − {offset} = {total}. Which integer is x?", expression, (coefficient, offset, total), solution

    def _build_groups_plus_remainder(self, variant: int):
        context, item = self._context(variant)
        groups, size, loose = 5 + variant, 8 + variant, 2 + variant
        expression = f"{groups} * {size} + {loose}"
        return f"A count in {context} has {groups} full containers with {size} {item} each and {loose} additional {item}. What is the total?", expression, (groups, size, loose), groups * size + loose

    def _build_arithmetic_sequence_sum(self, variant: int):
        context, item = self._context(variant)
        terms, first, difference = 5 + variant, 3 + variant, 2 + variant % 4
        expression = f"{terms} * (2 * {first} + ({terms} - 1) * {difference}) / 2"
        answer = terms * (2 * first + (terms - 1) * difference) // 2
        return f"A display in {context} has {terms} rows of {item}; the first has {first}, and every later row has {difference} more than the preceding row. How many are displayed?", expression, (terms, first, difference), answer

    def _build_complement_count(self, variant: int):
        context, item = self._context(variant)
        first, second, neither = 20 + variant * 2, 13 + variant, 11 + variant * 2
        total = first + second + neither
        expression = f"{total} - {first} - {second}"
        return f"Of {total} {item} reviewed in {context}, {first} were assigned to group A and {second} to group B, with no overlap. How many were assigned to neither group?", expression, (total, first, second), neither

    def _build_map_scale(self, variant: int):
        context, item = self._context(variant)
        map_units, actual_per_unit, target_units = 2 + variant % 4, 6 + variant, 7 + variant
        actual, answer = map_units * actual_per_unit, target_units * actual_per_unit
        expression = f"{actual} / {map_units} * {target_units}"
        return f"A planning map in {context} shows that {map_units} centimeters represents {actual} kilometers for a route carrying {item}. How many kilometers does {target_units} centimeters represent?", expression, (map_units, actual, target_units), answer

    def _build_combined_rectangle_area(self, variant: int):
        context, item = self._context(variant)
        first_length, first_width = 6 + variant, 4 + variant % 5
        second_length, second_width = 3 + variant, 2 + variant % 4
        expression = f"{first_length} * {first_width} + {second_length} * {second_width}"
        answer = first_length * first_width + second_length * second_width
        return f"Two nonoverlapping rectangular zones in {context} for {item} measure {first_length} by {first_width} units and {second_length} by {second_width} units. What is their combined area?", expression, (first_length, first_width, second_length, second_width), answer
