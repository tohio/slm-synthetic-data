from __future__ import annotations

from slm_synth.pretrain.artifacts.base import GroundedArtifact


class EducationalQAMCQMathArtifactFactory:
    """Create a finite catalog of verified, structurally distinct math MCQs."""

    SPECS = (
        (
            "expression_precedence",
            "What is the value of (18 + 7) × 4 − 9?",
            "(18 + 7) * 4 - 9",
            ("18", "7", "4", "9"),
            91,
        ),
        (
            "missing_factor",
            "Which integer replaces the question mark in 6 × ? + 5 = 77?",
            "(77 - 5) / 6",
            ("6", "5", "77"),
            12,
        ),
        (
            "equal_sharing",
            "A shipment of 156 notebooks is shared equally among 12 classrooms. How many notebooks does each classroom receive?",
            "156 / 12",
            ("156", "12"),
            13,
        ),
        (
            "two_step_remaining",
            "A supply room starts with 130 folders, sends out 24, and later sends out 37 more. How many folders remain?",
            "130 - 24 - 37",
            ("130", "24", "37"),
            69,
        ),
        (
            "expression_comparison",
            "Which is the largest value: 17 × 4, 29 + 35, or 90 − 19?",
            "90 - 19",
            ("17", "4", "29", "35", "90", "19"),
            71,
        ),
        (
            "fraction_of_quantity",
            "A collection contains 140 cards. If 3/5 of them are blue, how many blue cards are there?",
            "140 * 3 / 5",
            ("140", "3", "5"),
            84,
        ),
        (
            "proportional_scaling",
            "Four identical kits require 28 labels in total. At the same rate, how many labels are required for 9 kits?",
            "28 / 4 * 9",
            ("4", "28", "9"),
            63,
        ),
        (
            "rectangle_area",
            "What is the area of a rectangle with length 14 units and width 9 units?",
            "14 * 9",
            ("14", "9"),
            126,
        ),
        (
            "rectangle_perimeter",
            "What is the perimeter of a rectangle with length 18 units and width 7 units?",
            "2 * (18 + 7)",
            ("18", "7"),
            50,
        ),
        (
            "arithmetic_mean",
            "What is the arithmetic mean of 18, 22, 26, and 30?",
            "(18 + 22 + 26 + 30) / 4",
            ("18", "22", "26", "30"),
            24,
        ),
        (
            "total_from_mean",
            "6 measurements have an arithmetic mean of 17. What is their total?",
            "6 * 17",
            ("6", "17"),
            102,
        ),
        (
            "data_range",
            "What is the range of the values 12, 31, 19, and 27?",
            "31 - 12",
            ("12", "31", "19", "27"),
            19,
        ),
        (
            "combined_duration",
            "A setup takes 35 minutes and the activity takes 48 minutes. How many minutes do they take altogether?",
            "35 + 48",
            ("35", "48"),
            83,
        ),
        (
            "metric_conversion",
            "A ribbon is 7 meters long. Using 1 meter = 100 centimeters, how long is the ribbon in centimeters?",
            "7 * 100",
            ("7", "1", "100"),
            700,
        ),
        (
            "percentage_of_quantity",
            "What is 25% of 240?",
            "240 * 25 / 100",
            ("25", "240"),
            60,
        ),
        (
            "percentage_discount",
            "An item costs 160 dollars before a 15% discount. What is its price after the discount?",
            "160 - 160 * 15 / 100",
            ("160", "15"),
            136,
        ),
        (
            "percentage_increase",
            "A quantity of 80 is increased by 25%. What is the new quantity?",
            "80 + 80 * 25 / 100",
            ("80", "25"),
            100,
        ),
        (
            "two_rate_total",
            "A machine produces 14 parts per hour for 6 hours, then 19 parts per hour for 4 hours. How many parts does it produce in total?",
            "14 * 6 + 19 * 4",
            ("14", "6", "19", "4"),
            160,
        ),
        (
            "linear_equation",
            "Which integer satisfies 5x − 8 = 67?",
            "(67 + 8) / 5",
            ("5", "8", "67"),
            15,
        ),
        (
            "groups_plus_remainder",
            "A warehouse has 9 full trays with 13 items on each tray and 7 additional items. How many items are there altogether?",
            "9 * 13 + 7",
            ("9", "13", "7"),
            124,
        ),
        (
            "arithmetic_sequence_sum",
            "A display has 8 rows. The first row has 5 cards, and each later row has 3 more cards than the preceding row. How many cards are displayed?",
            "8 * (2 * 5 + 7 * 3) / 2",
            ("8", "5", "3"),
            124,
        ),
        (
            "complement_count",
            "Of 90 survey responses, 38 chose option A and 27 chose option B. How many chose neither option?",
            "90 - 38 - 27",
            ("90", "38", "27"),
            25,
        ),
        (
            "map_scale",
            "On a map, 3 centimeters represents 24 kilometers. At the same scale, how many kilometers does 9 centimeters represent?",
            "24 / 3 * 9",
            ("3", "24", "9"),
            72,
        ),
        (
            "combined_rectangle_area",
            "Two nonoverlapping rectangles measure 8 by 5 units and 6 by 4 units. What is their combined area?",
            "8 * 5 + 6 * 4",
            ("8", "5", "6", "4"),
            64,
        ),
    )

    FAMILIES = tuple(spec[0] for spec in SPECS)
    UNIQUE_CANDIDATE_CAPACITY = len(SPECS)

    @staticmethod
    def _choices(answer: int, index: int) -> list[str]:
        delta = max(2, min(20, max(1, abs(answer) // 10)))
        candidates = (answer - delta, answer + delta, answer + 2 * delta, answer - 2 * delta, answer + 1)
        distractors: list[str] = []
        for value in candidates:
            text = str(value)
            if value != answer and text not in distractors:
                distractors.append(text)
            if len(distractors) == 3:
                break
        choices = distractors
        choices.insert(index % 4, str(answer))
        return choices

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        if not 0 <= index < self.UNIQUE_CANDIDATE_CAPACITY:
            raise ValueError(
                f"educational_qa_mcq_math index {index} exceeds unique candidate capacity "
                f"{self.UNIQUE_CANDIDATE_CAPACITY}"
            )
        family, question, expression, numbers, answer = self.SPECS[index]
        choices = self._choices(answer, index)
        return GroundedArtifact(
            signal="educational_qa_mcq_math",
            family=family,
            artifact_id=f"educational_qa_mcq_math_{family}_{index + 1:09d}",
            payload={
                "question": question,
                "required_numeric_literals": list(numbers),
                "choices": choices,
                "answer": str(answer),
                "correct_index": choices.index(str(answer)),
                "expression": expression,
            },
        )
