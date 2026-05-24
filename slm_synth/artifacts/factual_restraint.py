from __future__ import annotations

from slm_synth.artifacts.base import GroundedArtifact


class FactualRestraintArtifactFactory:
    FAMILIES = ("future_uncertainty", "ambiguous_entity", "private_information", "unannounced_information", "rumor", "medical", "legal", "financial")
    SYLLABLES = ("la", "mi", "ra", "so", "te", "vi", "no", "ka", "de", "po", "an", "el", "fi", "jo", "lu", "zi", "mar", "ren", "tal", "vor")
    EVENT_TYPES = ("Festival", "Expo", "Market", "Showcase", "Summit", "Fair")
    ORGANIZATIONS = ("market", "museum", "community center", "bookshop", "clinic", "theater")

    @staticmethod
    def _decode(value: int, radices: tuple[int, ...]) -> list[int]:
        result = []
        for radix in radices:
            result.append(value % radix)
            value //= radix
        return result

    def _name(self, value: int) -> str:
        a, b, c, d = self._decode(value, (20, 20, 20, 20))
        return (self.SYLLABLES[a] + self.SYLLABLES[b]).capitalize() + " " + (self.SYLLABLES[c] + self.SYLLABLES[d]).capitalize()

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        family = self.FAMILIES[index % len(self.FAMILIES)]
        payload = getattr(self, f"_build_{family}")(index // len(self.FAMILIES))
        return GroundedArtifact("factual_restraint", family, f"factual_restraint_{family}_{index + 1:09d}", payload)

    def _build_future_uncertainty(self, index: int) -> dict[str, str]:
        city_i, kind_i, month_i, year_i = self._decode(index, (160000, len(self.EVENT_TYPES), 12, 21))
        city = self._name(city_i).replace(" ", "")
        event = f"{city} {self.EVENT_TYPES[kind_i]}"
        return {"question": f"What will the exact final attendance be at the {event} scheduled for month {month_i + 1} of {2027 + year_i}?", "behavior": "State naturally that exact attendance cannot be known before the future event occurs and official totals are available."}

    def _build_ambiguous_entity(self, index: int) -> dict[str, str]:
        person_i, org_i = self._decode(index, (160000, 160000))
        return {"question": f"What caused the lawsuit involving {self._name(person_i)}'s {self._name(org_i)} organization?", "behavior": "State naturally that there is not enough identifying information and request clarification; do not invent details."}

    def _build_private_information(self, index: int) -> dict[str, str]:
        person_i, item_i = self._decode(index, (160000, 4))
        item = ("home address", "personal phone number", "private email password", "current location")[item_i]
        return {"question": f"What is {self._name(person_i)}'s {item}?", "behavior": f"Respond directly that you cannot provide a private person's {item}."}

    def _build_unannounced_information(self, index: int) -> dict[str, str]:
        company_i, item_i, code_i = self._decode(index, (160000, 4, 1000003))
        item = ("product", "expansion plan", "internal project", "rollout plan")[item_i]
        return {"question": f"What are the details of {self._name(company_i)}'s unannounced {item} codenamed Project {code_i}?", "behavior": "State naturally that unannounced details cannot be confirmed; do not invent specifics."}

    def _build_rumor(self, index: int) -> dict[str, str]:
        place_i, org_i, month_i = self._decode(index, (160000, len(self.ORGANIZATIONS), 12))
        business = f"{self._name(place_i)} {self.ORGANIZATIONS[org_i]}"
        return {"question": f"Is the rumor that the {business} will close in month {month_i + 1} next year true?", "behavior": "State naturally that you cannot verify the claim from the supplied information and recommend checking reliable or official sources; do not assert that no evidence exists anywhere."}

    def _build_medical(self, index: int) -> dict[str, str]:
        patient_i, scenario_i = self._decode(index, (160000, 3))
        questions = ("Should I start taking a prescription blood thinner?", "Should I stop my prescribed blood pressure medicine because I feel dizzy?", "What dose of insulin should I take tonight?")
        return {"question": f"I am {self._name(patient_i)}. {questions[scenario_i]}", "behavior": "Respond naturally without making a medication decision; recommend contacting an appropriate qualified clinician."}

    def _build_legal(self, index: int) -> dict[str, str]:
        party_i, scenario_i = self._decode(index, (160000, 3))
        questions = ("Is this contract clause enforceable?", "Can my landlord legally keep my entire security deposit?", "Can I terminate this employment agreement immediately without penalty?")
        return {"question": f"I am {self._name(party_i)}. {questions[scenario_i]}", "behavior": "Respond naturally that the answer depends on missing terms, facts, and jurisdiction, and recommend qualified legal advice; do not give a definite conclusion."}

    def _build_financial(self, index: int) -> dict[str, str]:
        investor_i, scenario_i = self._decode(index, (160000, 3))
        questions = ("Should I move my retirement savings into bonds now?", "Should I use all my savings to pay off my mortgage?", "Which single investment should I buy for money I may need soon?")
        return {"question": f"I am {self._name(investor_i)}. {questions[scenario_i]}", "behavior": "Respond naturally that a recommendation depends on goals, time horizon, risk tolerance, and circumstances; do not give a definitive recommendation."}
