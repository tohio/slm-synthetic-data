from __future__ import annotations

from slm_synth.artifacts.base import GroundedArtifact
from slm_synth.artifacts.lexicon import CITIES, FIRST_NAMES, LAST_NAMES, ORGANIZATION_NAMES, VENUES, full_name


class EducationalQAMCQGeneralArtifactFactory:
    """Create deterministic, natural evidence-grounded MCQs."""

    FAMILIES = ("python_behavior", "grammar", "vocabulary", "reading", "fictional_rule", "policy", "scientific_method", "ordering")
    PLACES = ("blue drawer", "green cabinet", "front desk", "storage shelf", "toolbox", "library cart", "archive bin", "side locker", "supply room", "service counter", "upper cabinet", "lower tray")
    OBJECTS = ("spare key", "signed forms", "visitor badge", "survey folder", "repair manual", "supply list", "delivery slip", "permit copy", "tool record", "inventory sheet", "registration card", "inspection note")
    ADVERBS = ("quietly", "carefully", "quickly", "patiently", "neatly", "calmly", "slowly", "briefly", "softly", "promptly")
    ADJECTIVE_CONTEXT = {
        "fragile": ("easily broken", "The item needed protective wrapping."),
        "narrow": ("not wide", "Only one person could pass through at a time."),
        "ancient": ("very old", "The object was preserved from centuries ago."),
        "bright": ("giving much light", "It illuminated the nearby shelf."),
        "careful": ("taking caution", "No details were overlooked."),
        "silent": ("not making sound", "No one heard it move."),
        "scarce": ("hard to find", "Only a few remained in stock."),
        "durable": ("long-lasting", "It survived repeated daily use."),
        "precise": ("exact", "Every measurement matched the specification."),
        "modest": ("not excessive", "The request required only a small amount."),
    }
    VERBS = ("stored", "placed", "carried", "moved", "checked", "returned", "sealed", "organized", "recorded", "reviewed")
    FICTIONAL_REGIONS = ("Orin", "Pelin", "Nareth", "Volar", "Kesra", "Tavon", "Merin", "Solara", "Bren", "Doria", "Calen", "Ivara", "Aster", "Belin", "Corin", "Elora", "Faron", "Galen", "Haven", "Joren")
    FICTIONAL_LABELS = ("Velora", "Moro", "Sorin", "Kessa", "Tavin", "Liora", "Brelan", "Norvi", "Selka", "Dorin", "Pavri", "Merola", "Quorin", "Vessa", "Talune", "Rovik")
    MARK_COLORS = ("silver", "blue", "red", "amber", "green", "white", "black", "copper")
    MARK_SHAPES = ("handle", "stripe", "circle", "flag", "seal", "ribbon", "triangle", "star")
    DEPARTMENTS = ("records", "finance", "research", "facilities", "archive", "security", "clinical", "procurement")
    VARIABLES = ("type of music", "amount of light", "water temperature", "type of soil", "room temperature", "amount of water", "container size", "time exposed")

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
        return GroundedArtifact("educational_qa_mcq_general", family, f"educational_qa_mcq_general_{family}_{index + 1:09d}", payload)

    @staticmethod
    def _record(evidence: str, question: str, choices: list[str], answer: str) -> dict[str, object]:
        return {"evidence": evidence, "question": question, "choices": choices, "correct_index": choices.index(answer), "answer": answer}

    def _build_python_behavior(self, index: int) -> dict[str, object]:
        repeats = 2 + (index % 5)
        first = 10 + index * 2
        second = first + 1
        values = [first] * repeats + [second]
        evidence = f"values = {values}\nresult = values.count({first})"
        choices = [str(repeats - 1), str(repeats), str(repeats + 1), "0"]
        return self._record(evidence, "What is the final value of result?", choices, str(repeats))

    def _build_grammar(self, index: int) -> dict[str, object]:
        person_i, adv_i, obj_i, verb_i = self._decode(index, (len(FIRST_NAMES) * len(LAST_NAMES), len(self.ADVERBS), len(self.OBJECTS), len(self.VERBS)))
        name, adv, obj, verb = full_name(person_i), self.ADVERBS[adv_i], self.OBJECTS[obj_i], self.VERBS[verb_i]
        evidence = f'Sentence: "{name} {adv} {verb} the {obj}."'
        return self._record(evidence, "Which word is an adverb?", [name, adv, verb, obj], adv)

    def _build_vocabulary(self, index: int) -> dict[str, object]:
        adjective_i, person_i, obj_i, place_i = self._decode(index, (len(self.ADJECTIVE_CONTEXT), len(FIRST_NAMES) * len(LAST_NAMES), len(self.OBJECTS), len(self.PLACES)))
        adjective = tuple(self.ADJECTIVE_CONTEXT)[adjective_i]
        answer, clue = self.ADJECTIVE_CONTEXT[adjective]
        name, obj, place = full_name(person_i), self.OBJECTS[obj_i], self.PLACES[place_i]
        evidence = f'Sentence: "At the {place}, {name} noticed the {obj} was {adjective}. {clue}"'
        return self._record(evidence, f"What does {adjective} mean in the sentence?", [answer, "very loud", "unrelated", "very fast"], answer)

    def _build_reading(self, index: int) -> dict[str, object]:
        person_i, place_i, object_i, later_i = self._decode(index, (len(FIRST_NAMES) * len(LAST_NAMES), len(self.PLACES), len(self.OBJECTS), len(self.PLACES)))
        name, place, obj, later = full_name(person_i), self.PLACES[place_i], self.OBJECTS[object_i], self.PLACES[later_i]
        if later == place:
            later = self.PLACES[(later_i + 1) % len(self.PLACES)]
        alternatives = [candidate for candidate in self.PLACES if candidate not in {place, later}][:3]
        evidence = f"Passage: {name} stored the {obj} in the {place}. Later, {name} visited the {later}."
        return self._record(evidence, f"Where did {name} store the {obj}?", [f"in the {place}"] + [f"in the {candidate}" for candidate in alternatives], f"in the {place}")

    def _build_fictional_rule(self, index: int) -> dict[str, object]:
        region_i, label_i, object_i, color_i, shape_i = self._decode(index, (len(self.FICTIONAL_REGIONS), len(self.FICTIONAL_LABELS), len(self.OBJECTS), len(self.MARK_COLORS), len(self.MARK_SHAPES)))
        region, label, obj = self.FICTIONAL_REGIONS[region_i], self.FICTIONAL_LABELS[label_i], self.OBJECTS[object_i]
        mark = f"{self.MARK_COLORS[color_i]} {self.MARK_SHAPES[shape_i]}"
        evidence = f'Rule: In {region}, any item marked with a {mark} is called a "{label}." The {obj} is marked with a {mark}.'
        choices = [label] + [self.FICTIONAL_LABELS[(label_i + offset) % len(self.FICTIONAL_LABELS)] for offset in (1, 2, 3)]
        return self._record(evidence, f"Which fictional label applies to the {obj}?", choices, label)

    def _build_policy(self, index: int) -> dict[str, object]:
        obj_i, place_i, dept_i, city_i, org_i, venue_i = self._decode(index, (len(self.OBJECTS), len(self.PLACES), len(self.DEPARTMENTS), len(CITIES), len(ORGANIZATION_NAMES), len(VENUES)))
        obj, place, dept, city = self.OBJECTS[obj_i], self.PLACES[place_i], self.DEPARTMENTS[dept_i], CITIES[city_i]
        organization, venue = ORGANIZATION_NAMES[org_i], VENUES[venue_i]
        evidence = f"Policy: At {organization} in {city}, only approved {dept} employees may place the {obj} in the {place} at the {venue}."
        violation = f"An unapproved visitor places the {obj} in the {place}."
        choices = [f"An approved {dept} employee places the {obj} in the {place}.", violation, f"A visitor waits in the {city} reception area.", f"A {dept} employee checks approval status first."]
        return self._record(evidence, "Which action violates the policy?", choices, violation)

    def _build_scientific_method(self, index: int) -> dict[str, object]:
        variable_i, item_i, site_i, person_i = self._decode(index, (len(self.VARIABLES), len(self.OBJECTS), len(self.PLACES), len(FIRST_NAMES) * len(LAST_NAMES)))
        variable, site, researcher = self.VARIABLES[variable_i], self.PLACES[site_i], full_name(person_i)
        evidence = f"Experiment: At the {site}, {researcher} gives Group A and Group B the same {self.OBJECTS[item_i]}, instructions, and observation method. The groups differ only in {variable}."
        choices = [variable, "observation method", "instructions", "item tested"]
        return self._record(evidence, "Which variable was deliberately changed?", choices, variable)

    def _build_ordering(self, index: int) -> dict[str, object]:
        permutations = len(self.FICTIONAL_LABELS) * (len(self.FICTIONAL_LABELS) - 1) * (len(self.FICTIONAL_LABELS) - 2) * (len(self.FICTIONAL_LABELS) - 3)
        order_i, city_i, venue_i = self._decode(index, (permutations, len(CITIES), len(VENUES)))
        available = list(self.FICTIONAL_LABELS)
        labels: list[str] = []
        for remaining in (len(available), len(available) - 1, len(available) - 2, len(available) - 3):
            position = order_i % remaining
            order_i //= remaining
            labels.append(available.pop(position))
        evidence = f"Ordering rule: For the display at the {VENUES[venue_i]} in {CITIES[city_i]}, {labels[0]} before {labels[1]}, {labels[1]} before {labels[2]}, and {labels[2]} before {labels[3]}."
        return self._record(evidence, "Which item comes first?", labels, labels[0])
