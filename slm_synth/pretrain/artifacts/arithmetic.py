from __future__ import annotations

import ast

from slm_synth.pretrain.artifacts.base import GroundedArtifact


class ArithmeticArtifactFactory:
    """Create finite, structurally distinct, locally verified arithmetic tasks.

    Early indices rotate through materially different reasoning families and
    semantic contexts before changing operands. A smoke run therefore does not
    spend requests on copies of one sentence frame with different numbers.
    """

    FAMILIES = (
        "direct_expression",
        "missing_start_after_increase",
        "missing_start_after_decrease",
        "two_step_remaining",
        "gain_then_spend",
        "exact_group_count",
        "equal_share_size",
        "groups_with_loose_items",
        "compare_group_totals",
        "target_gap",
        "three_source_total",
        "constant_rate_total",
        "two_rate_total",
        "known_portion_equal_shares",
        "net_change",
        "rectangle_perimeter",
    )

    CONTEXTS = (
        ("archive inventory", "document folders", "inventory sheet"),
        ("community garden", "seed packets", "planting log"),
        ("repair workshop", "replacement gears", "service ledger"),
        ("science laboratory", "sample vials", "experiment log"),
        ("school library", "returned books", "circulation report"),
        ("food pantry", "meal kits", "distribution record"),
        ("theater office", "admission tickets", "booking report"),
        ("shipping depot", "sealed parcels", "dispatch manifest"),
        ("wildlife clinic", "supply packs", "treatment log"),
        ("museum storeroom", "display labels", "catalog record"),
        ("cycling event", "water bottles", "course checklist"),
        ("music program", "practice booklets", "rehearsal record"),
        ("makerspace", "component trays", "build worksheet"),
        ("language center", "lesson cards", "session report"),
        ("field survey", "observation forms", "survey notebook"),
        ("bakery kitchen", "bread rolls", "production sheet"),
        ("transit office", "route notices", "operations log"),
        ("emergency shelter", "blanket bundles", "intake record"),
    )

    DIRECT_EXPRESSION_SHAPES = (
        "({a} + {b}) * {c} - {d}",
        "{a} * {c} + {b} - {d}",
        "{a} + {b} * {c} - {d}",
        "({a} - {b}) * {c} + {d}",
        "{a} * {c} - ({b} + {d})",
        "{a} - ({b} - {c}) * {d}",
        "{a} * ({b} - {c}) + {d}",
        "{a} + {b} + {c} * {d}",
        "{a} * {b} + {c} + {d}",
        "({a} + {b} + {c}) * {d}",
        "{a} * {b} - {c} * {d}",
        "{a} * ({b} + {c}) - {d}",
        "{a} + {b} * ({c} + {d})",
        "({a} - {b} + {c}) * {d}",
        "{a} + ({b} - {c}) * {d}",
        "({a} + {c}) * ({b} - {d})",
        "{a} * {b} + {c} * {d}",
        "({a} + {b}) * {c} + {d}",
    )
    UNIQUE_CANDIDATE_CAPACITY = len(FAMILIES) * len(CONTEXTS)

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
        payload = getattr(self, f"_build_{family}")(family_index)
        return GroundedArtifact(
            signal="arithmetic",
            family=family,
            artifact_id=f"arithmetic_{family}_{index + 1:09d}",
            payload=payload,
        )

    def _context(self, index: int) -> tuple[str, str, str, int]:
        context_id = index % len(self.CONTEXTS)
        operand_variant = index
        domain, item, source = self.CONTEXTS[context_id]
        return domain, item, source, operand_variant

    @staticmethod
    def _word_payload(
        *,
        instruction: str,
        domain: str,
        item: str,
        source: str,
        facts: list[str],
        expression: str,
        numeric_literals: list[int],
        answer: int,
    ) -> dict[str, object]:
        return {
            "instruction": instruction,
            "domain": domain,
            "item": item,
            "source": source,
            "facts": facts,
            "expression": expression,
            "required_numeric_literals": [str(value) for value in numeric_literals],
            "required_text_literals": [domain, item, source],
            "answer": str(answer),
        }

    @staticmethod
    def _evaluate(expression: str) -> int:
        tree = ast.parse(expression, mode="eval")

        def visit(node: ast.AST) -> int:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                return node.value
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
            raise ValueError("unsupported arithmetic expression")

        return visit(tree)

    def _build_direct_expression(self, index: int) -> dict[str, object]:
        shape_id = index % len(self.DIRECT_EXPRESSION_SHAPES)
        operand_variant = index
        a = 24 + operand_variant % 37
        b = 13 + (operand_variant * 5) % 17
        c = 2 + (operand_variant * 3) % 7
        d = 1 + (operand_variant * 7) % 8
        expression = self.DIRECT_EXPRESSION_SHAPES[shape_id].format(a=a, b=b, c=c, d=d)
        return {
            "instruction": "Create a direct expression-evaluation question preserving parentheses and precedence.",
            "expression": expression,
            "required_numeric_literals": [str(a), str(b), str(c), str(d)],
            "answer": str(self._evaluate(expression)),
        }

    def _build_missing_start_after_increase(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        added, start = 13 + variant % 41, 31 + (variant * 7) % 89
        final = start + added
        return self._word_payload(
            instruction="Ask for the unknown starting quantity before a recorded increase.",
            domain=domain, item=item, source=source,
            facts=[f"{added} were added", f"the recorded final quantity is {final}"],
            expression=f"{final} - {added}", numeric_literals=[added, final], answer=start,
        )

    def _build_missing_start_after_decrease(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        removed, final = 9 + variant % 37, 28 + (variant * 11) % 83
        return self._word_payload(
            instruction="Ask for the unknown starting quantity before a recorded decrease.",
            domain=domain, item=item, source=source,
            facts=[f"{removed} were removed", f"the recorded final quantity is {final}"],
            expression=f"{final} + {removed}", numeric_literals=[removed, final], answer=final + removed,
        )

    def _build_two_step_remaining(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        first, second, remaining = 11 + variant % 29, 7 + (variant * 3) % 23, 35 + (variant * 5) % 71
        start = first + second + remaining
        return self._word_payload(
            instruction="Ask for the quantity remaining after two separately recorded removals.",
            domain=domain, item=item, source=source,
            facts=[f"the starting quantity is {start}", f"first remove {first}", f"then remove {second}"],
            expression=f"{start} - {first} - {second}", numeric_literals=[start, first, second], answer=remaining,
        )

    def _build_gain_then_spend(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        start, gained, used = 42 + variant % 67, 16 + (variant * 5) % 43, 5 + (variant * 2) % 9
        return self._word_payload(
            instruction="Ask for the final quantity after an increase followed by a decrease.",
            domain=domain, item=item, source=source,
            facts=[f"begin with {start}", f"receive {gained}", f"later use {used}"],
            expression=f"{start} + {gained} - {used}", numeric_literals=[start, gained, used], answer=start + gained - used,
        )

    def _build_exact_group_count(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        per_group, groups = 6 + variant % 19, 30 + (variant * 5) % 31
        total = per_group * groups
        return self._word_payload(
            instruction="Ask how many equal groups are required when total and group size are known.",
            domain=domain, item=item, source=source,
            facts=[f"the total is {total}", f"each group contains {per_group}"],
            expression=f"{total} / {per_group}", numeric_literals=[total, per_group], answer=groups,
        )

    def _build_equal_share_size(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        groups, share = 4 + variant % 8, 13 + (variant * 7) % 29
        total = groups * share
        return self._word_payload(
            instruction="Ask for each equal share when total and share count are known.",
            domain=domain, item=item, source=source,
            facts=[f"divide {total} equally", f"use {groups} shares"],
            expression=f"{total} / {groups}", numeric_literals=[total, groups], answer=share,
        )

    def _build_groups_with_loose_items(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        groups, per_group = 4 + variant % 19, 7 + (variant * 3) % 23
        loose = 1 + (variant * 5) % (per_group - 1)
        return self._word_payload(
            instruction="Ask for a total formed by full groups plus a separate loose quantity.",
            domain=domain, item=item, source=source,
            facts=[f"there are {groups} full groups", f"each group has {per_group}", f"there are {loose} loose items"],
            expression=f"{groups} * {per_group} + {loose}", numeric_literals=[groups, per_group, loose],
            answer=groups * per_group + loose,
        )

    def _build_compare_group_totals(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        left_groups, left_size = 8 + variant % 13, 12 + (variant * 3) % 17
        right_groups, right_size = 4 + (variant * 5) % 7, 6 + (variant * 7) % 11
        left_total, right_total = left_groups * left_size, right_groups * right_size
        return self._word_payload(
            instruction="Ask how many more items are represented by the larger of two grouped totals.",
            domain=domain, item=item, source=source,
            facts=[f"first total uses {left_groups} groups of {left_size}", f"second total uses {right_groups} groups of {right_size}"],
            expression=f"{left_groups} * {left_size} - {right_groups} * {right_size}",
            numeric_literals=[left_groups, left_size, right_groups, right_size], answer=left_total - right_total,
        )

    def _build_target_gap(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        current, gap = 45 + variant % 91, 12 + (variant * 7) % 53
        target = current + gap
        return self._word_payload(
            instruction="Ask how many more are needed to reach a target from a current count.",
            domain=domain, item=item, source=source,
            facts=[f"the current count is {current}", f"the target count is {target}"],
            expression=f"{target} - {current}", numeric_literals=[current, target], answer=gap,
        )

    def _build_three_source_total(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        first, second, third = 18 + variant % 47, 21 + (variant * 5) % 43, 16 + (variant * 7) % 41
        return self._word_payload(
            instruction="Ask for a combined total reported by three independent sources.",
            domain=domain, item=item, source=source,
            facts=[f"first source reports {first}", f"second source reports {second}", f"third source reports {third}"],
            expression=f"{first} + {second} + {third}", numeric_literals=[first, second, third], answer=first + second + third,
        )

    def _build_constant_rate_total(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        periods, rate = 4 + variant % 17, 9 + (variant * 5) % 31
        return self._word_payload(
            instruction="Ask for a total accumulated at a constant whole-number rate over several periods.",
            domain=domain, item=item, source=source,
            facts=[f"the rate is {rate} per period", f"the activity lasts {periods} periods"],
            expression=f"{rate} * {periods}", numeric_literals=[rate, periods], answer=rate * periods,
        )

    def _build_two_rate_total(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        first_periods, first_rate = 3 + variant % 9, 8 + (variant * 3) % 19
        second_periods, second_rate = 2 + (variant * 5) % 8, 7 + (variant * 7) % 17
        return self._word_payload(
            instruction="Ask for a combined total produced during two stages with different rates.",
            domain=domain, item=item, source=source,
            facts=[f"stage one lasts {first_periods} periods at {first_rate} per period", f"stage two lasts {second_periods} periods at {second_rate} per period"],
            expression=f"{first_periods} * {first_rate} + {second_periods} * {second_rate}",
            numeric_literals=[first_periods, first_rate, second_periods, second_rate],
            answer=first_periods * first_rate + second_periods * second_rate,
        )

    def _build_known_portion_equal_shares(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        groups, share, known = 3 + variant % 13, 7 + (variant * 5) % 23, 40 + (variant * 7) % 47
        total = groups * share + known
        return self._word_payload(
            instruction="Ask for each equal unknown share after a known portion is removed from a total.",
            domain=domain, item=item, source=source,
            facts=[f"the full total is {total}", f"a known portion contains {known}", f"the rest forms {groups} equal shares"],
            expression=f"({total} - {known}) / {groups}", numeric_literals=[total, known, groups], answer=share,
        )

    def _build_net_change(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        start = 51 + variant % 73
        first_gain, loss, second_gain = 12 + (variant * 3) % 31, 9 + (variant * 5) % 27, 7 + (variant * 7) % 23
        return self._word_payload(
            instruction="Ask for the final count after three chronologically ordered changes.",
            domain=domain, item=item, source=source,
            facts=[f"start at {start}", f"add {first_gain}", f"remove {loss}", f"add another {second_gain}"],
            expression=f"{start} + {first_gain} - {loss} + {second_gain}",
            numeric_literals=[start, first_gain, loss, second_gain], answer=start + first_gain - loss + second_gain,
        )

    def _build_rectangle_perimeter(self, index: int) -> dict[str, object]:
        domain, item, source, variant = self._context(index)
        length, width = 8 + variant % 29, 5 + (variant * 5) % 19
        return self._word_payload(
            instruction="Ask for the perimeter of a rectangular work area from its whole-number side lengths.",
            domain=domain, item=item, source=source,
            facts=[f"the length is {length} units", f"the width is {width} units"],
            expression=f"2 * ({length} + {width})", numeric_literals=[length, width], answer=2 * (length + width),
        )
