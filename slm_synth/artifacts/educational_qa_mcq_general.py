from __future__ import annotations

from slm_synth.artifacts.base import GroundedArtifact


class EducationalQAMCQGeneralArtifactFactory:
    """Create deterministic, evidence-grounded MCQs with large combinatorial spaces."""

    FAMILIES = ("python_behavior", "grammar", "vocabulary", "reading", "fictional_rule", "policy", "scientific_method", "ordering")
    SYLLABLES = ("la", "mi", "ra", "so", "te", "vi", "no", "ka", "de", "po", "an", "el", "fi", "jo", "lu", "zi", "mar", "ren", "tal", "vor")
    PLACES = ("blue drawer", "green cabinet", "front desk", "storage shelf", "toolbox", "library cart", "archive bin", "side locker", "supply room", "service counter", "upper cabinet", "lower tray")
    OBJECTS = ("spare key", "signed forms", "visitor badge", "survey folder", "repair manual", "supply list", "delivery slip", "permit copy", "tool record", "inventory sheet", "registration card", "inspection note")
    ADVERBS = ("quietly", "carefully", "quickly", "patiently", "neatly", "calmly", "slowly", "briefly", "softly", "promptly")
    ADJECTIVES = ("fragile", "narrow", "ancient", "bright", "careful", "silent", "scarce", "durable", "precise", "modest")
    VERBS = ("stored", "placed", "carried", "moved", "checked", "returned", "sealed", "organized", "recorded", "reviewed")

    @staticmethod
    def _decode(value: int, radices: tuple[int, ...]) -> list[int]:
        result: list[int] = []
        for radix in radices:
            result.append(value % radix)
            value //= radix
        return result

    def _name(self, value: int) -> str:
        a, b, c = self._decode(value, (len(self.SYLLABLES), len(self.SYLLABLES), len(self.SYLLABLES)))
        return (self.SYLLABLES[a] + self.SYLLABLES[b] + self.SYLLABLES[c]).capitalize()

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
        name_i, adv_i, obj_i, verb_i = self._decode(index, (8000, len(self.ADVERBS), len(self.OBJECTS), len(self.VERBS)))
        name, adv, obj, verb = self._name(name_i), self.ADVERBS[adv_i], self.OBJECTS[obj_i], self.VERBS[verb_i]
        evidence = f'Sentence: "{name} {adv} {verb} the {obj}."'
        return self._record(evidence, "Which word is an adverb?", [name, adv, verb, obj], adv)

    def _build_vocabulary(self, index: int) -> dict[str, object]:
        adjective_i, name_i, obj_i, place_i = self._decode(index, (len(self.ADJECTIVES), 8000, len(self.OBJECTS), len(self.PLACES)))
        adjective, name, obj, place = self.ADJECTIVES[adjective_i], self._name(name_i), self.OBJECTS[obj_i], self.PLACES[place_i]
        meanings = {"fragile": "easily broken", "narrow": "not wide", "ancient": "very old", "bright": "giving much light", "careful": "taking caution", "silent": "not making sound", "scarce": "hard to find", "durable": "long-lasting", "precise": "exact", "modest": "not excessive"}
        answer = meanings[adjective]
        evidence = f'Sentence: "At the {place}, {name} noticed the {obj} was {adjective}."'
        return self._record(evidence, f"What does {adjective} mean in the sentence?", [answer, "very loud", "unrelated", "very fast"], answer)

    def _build_reading(self, index: int) -> dict[str, object]:
        name_i, place_i, object_i, later_i = self._decode(index, (8000, len(self.PLACES), len(self.OBJECTS), len(self.PLACES)))
        name, place, obj, later = self._name(name_i), self.PLACES[place_i], self.OBJECTS[object_i], self.PLACES[later_i]
        if later == place:
            later = self.PLACES[(later_i + 1) % len(self.PLACES)]
        alternatives = [p for p in self.PLACES if p != place][:3]
        evidence = f"Passage: {name} stored the {obj} in the {place}. Later, {name} visited the {later}."
        return self._record(evidence, f"Where did {name} store the {obj}?", [f"in the {place}"] + [f"in the {p}" for p in alternatives], f"in the {place}")

    def _build_fictional_rule(self, index: int) -> dict[str, object]:
        mark_i, object_i, label_i, region_i = self._decode(index, (len(self.PLACES), len(self.OBJECTS), 8000, 8000))
        label, region = self._name(label_i), self._name(region_i)
        mark, obj = self.PLACES[mark_i], self.OBJECTS[object_i]
        evidence = f'Rule: In {region}, any item marked with a {mark} is called a "{label}." The {obj} is marked with a {mark}.'
        choices = [label, self._name((label_i + 1) % 8000), self._name((label_i + 2) % 8000), self._name((label_i + 3) % 8000)]
        return self._record(evidence, f"Which fictional label applies to the {obj}?", choices, label)

    def _build_policy(self, index: int) -> dict[str, object]:
        obj_i, place_i, dept_i, role_i = self._decode(index, (len(self.OBJECTS), len(self.PLACES), 8000, 8000))
        obj, place, dept, role = self.OBJECTS[obj_i], self.PLACES[place_i], self._name(dept_i), self._name(role_i)
        evidence = f"Policy: Only approved {dept} employees may place the {obj} in the {place}."
        violation = f"An unapproved {role} visitor places the {obj} in the {place}."
        choices = [f"An approved {dept} employee places the {obj} in the {place}.", violation, f"A {role} visitor waits outside.", f"A {dept} employee checks approval status first."]
        return self._record(evidence, "Which action violates the policy?", choices, violation)

    def _build_scientific_method(self, index: int) -> dict[str, object]:
        variable_i, item_i, site_i, name_i = self._decode(index, (8, len(self.OBJECTS), len(self.PLACES), 8000))
        variables = ("type of music", "amount of light", "water temperature", "type of soil", "room temperature", "amount of water", "container size", "time exposed")
        variable, site, researcher = variables[variable_i], self.PLACES[site_i], self._name(name_i)
        evidence = f"Experiment: At the {site}, {researcher} gives Group A and Group B the same {self.OBJECTS[item_i]}, instructions, and observation method. The groups differ only in {variable}."
        choices = [variable, "observation method", "instructions", "item tested"]
        return self._record(evidence, "Which variable was deliberately changed?", choices, variable)

    def _build_ordering(self, index: int) -> dict[str, object]:
        a_i, b_i, c_i, d_i = self._decode(index, (8000, 7999, 7997, 7993))
        labels = [self._name(a_i), self._name(b_i + 1), self._name(c_i + 2), self._name(d_i + 3)]
        if len(set(labels)) < 4:
            labels = [f"{label}{position}" for position, label in enumerate(labels, 1)]
        evidence = f"Ordering rule: {labels[0]} before {labels[1]}, {labels[1]} before {labels[2]}, and {labels[2]} before {labels[3]}."
        return self._record(evidence, "Which item comes first?", labels, labels[0])
