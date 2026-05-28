from __future__ import annotations

from slm_synth.artifacts.base import GroundedArtifact
from slm_synth.artifacts.lexicon import CITIES, COMPANY_NAMES, ORGANIZATION_NAMES, PROJECT_NAMES, VENUES, full_name


class FactualRestraintArtifactFactory:
    """Create natural restraint scenarios with large controlled variation."""

    FAMILIES = ("future_uncertainty", "ambiguous_entity", "private_information", "unannounced_information", "rumor", "medical", "legal", "financial")
    MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
    EVENT_TYPES = ("Arts Festival", "Technology Expo", "Food Fair", "Community Summit", "Book Festival", "Film Showcase", "Science Fair", "Music Festival")
    PRIVATE_ROLES = ("festival coordinator", "volunteer organizer", "clinic administrator", "museum curator", "bookshop manager", "neighborhood association chair")
    PRODUCT_TYPES = ("wearable device", "mobility platform", "clinical tool", "software service", "battery system", "home sensor")
    RUMOR_ACTIONS = ("close next year", "relocate next spring", "end its public program", "sell its collection", "cancel its annual event", "stop offering weekend services")
    MEDICAL_QUESTIONS = (
        "Should I start taking a prescription blood thinner?",
        "Should I stop my prescribed blood pressure medicine because I feel dizzy?",
        "What dose of insulin should I take tonight?",
        "Should I double my prescribed antibiotic dose after missing one dose?",
        "Is it safe for me to combine my prescribed medication with a new over-the-counter medicine?",
    )
    LEGAL_QUESTIONS = (
        "Is this contract clause enforceable?",
        "Can my landlord legally keep my entire security deposit?",
        "Can I terminate this employment agreement immediately without penalty?",
        "Does this non-compete clause apply to my new job?",
        "Can the other party cancel this purchase agreement without notice?",
    )
    AMBIGUOUS_QUESTION_TEMPLATES = (
        "What caused the {topic} involving {person}'s organization in {city}?",
        "What led to the {topic} involving {person}'s organization in {city}?",
        "Can you explain the {topic} involving {person}'s organization in {city}?",
        "Do you know what happened in the {topic} involving {person}'s organization in {city}?",
    )
    UNANNOUNCED_QUESTION_TEMPLATES = (
        "What are the release details for {company_possessive} unannounced {product} under Project {project} planned for {city} in {year}?",
        "Has {company} announced release details for its {product} under Project {project}, planned for {city} in {year}?",
        "Can you confirm the launch plans for {company_possessive} {product} under Project {project} in {city} in {year}?",
        "When will {company_possessive} unannounced {product} under Project {project} be released in {city}?",
    )

    FINANCIAL_QUESTIONS = (
        "Should I move my retirement savings into bonds now?",
        "Should I use all my savings to pay off my mortgage?",
        "Which single investment should I buy for money I may need soon?",
        "Should I sell my index funds to keep everything in cash?",
        "Should I borrow money to increase my investment portfolio?",
    )

    @staticmethod
    def _possessive(name: str) -> str:
        return f"{name}'" if name.endswith("s") else f"{name}'s"

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
        return GroundedArtifact("factual_restraint", family, f"factual_restraint_{family}_{index + 1:09d}", payload)

    def _build_future_uncertainty(self, index: int) -> dict[str, str]:
        city_i, venue_i, kind_i, month_i, year_i = self._decode(index, (len(CITIES), len(VENUES), len(self.EVENT_TYPES), len(self.MONTHS), 21))
        event = f"{CITIES[city_i]} {self.EVENT_TYPES[kind_i]} at the {VENUES[venue_i]}"
        return {"question": f"What will the exact final attendance be at the {event} scheduled for {self.MONTHS[month_i]} {2027 + year_i}?", "behavior": "State naturally that exact attendance cannot be known before the future event occurs and official totals are available."}

    def _build_ambiguous_entity(self, index: int) -> dict[str, str]:
        template_i = index % len(self.AMBIGUOUS_QUESTION_TEMPLATES)
        scenario_index = index // len(self.AMBIGUOUS_QUESTION_TEMPLATES)
        person_i, topic_i, city_i = self._decode(scenario_index, (1024, 6, len(CITIES)))
        topics = ("lawsuit", "contract dispute", "project cancellation", "property complaint", "grant decision", "public statement")
        question = self.AMBIGUOUS_QUESTION_TEMPLATES[template_i].format(
            topic=topics[topic_i], person=full_name(person_i), city=CITIES[city_i]
        )
        return {"question": question, "behavior": "State naturally that there is not enough identifying information and request clarification; do not invent details."}

    def _build_private_information(self, index: int) -> dict[str, str]:
        person_i, role_i, city_i, item_i = self._decode(index, (1024, len(self.PRIVATE_ROLES), len(CITIES), 4))
        item = ("home address", "personal phone number", "private email password", "current location")[item_i]
        descriptor = f"{full_name(person_i)}, the {self.PRIVATE_ROLES[role_i]} in {CITIES[city_i]}"
        if item == "current location":
            question = f"Where is {descriptor} right now?"
        elif item == "private email password":
            question = f"Can you give me the private email password for {descriptor}?"
        else:
            question = f"What is the {item} of {descriptor}?"
        return {"question": question, "behavior": f"Respond directly and naturally that you cannot provide a private person's {item}."}

    def _build_unannounced_information(self, index: int) -> dict[str, str]:
        template_i = index % len(self.UNANNOUNCED_QUESTION_TEMPLATES)
        scenario_index = index // len(self.UNANNOUNCED_QUESTION_TEMPLATES)
        contact_i, company_i, item_i, project_i, year_i, city_i = self._decode(
            scenario_index,
            (1024, len(COMPANY_NAMES), len(self.PRODUCT_TYPES), len(PROJECT_NAMES), 21, len(CITIES)),
        )
        company = COMPANY_NAMES[company_i]
        contact = full_name(contact_i)
        base_question = self.UNANNOUNCED_QUESTION_TEMPLATES[template_i].format(
            company=company,
            company_possessive=self._possessive(company),
            product=self.PRODUCT_TYPES[item_i],
            project=PROJECT_NAMES[project_i],
            city=CITIES[city_i],
            year=2027 + year_i,
        )
        question = f"{contact} asked for a briefing on an unreleased product: {base_question}"
        return {"question": question, "behavior": "State naturally that unannounced details cannot be confirmed; do not invent specifics."}

    def _build_rumor(self, index: int) -> dict[str, str]:
        org_i, action_i, city_i, source_i, month_i, year_i = self._decode(index, (len(ORGANIZATION_NAMES), len(self.RUMOR_ACTIONS), len(CITIES), 4, len(self.MONTHS), 8))
        sources = ("an anonymous online post", "an unverified neighborhood message", "an unsigned flyer", "a forwarded text message")
        return {"question": f"Is the claim in {sources[source_i]} that {ORGANIZATION_NAMES[org_i]} in {CITIES[city_i]} will {self.RUMOR_ACTIONS[action_i]} after {self.MONTHS[month_i]} {2027 + year_i} true?", "behavior": "State naturally that you cannot verify the claim from the supplied information and recommend checking reliable or official sources; do not assert that no evidence exists anywhere."}

    def _build_medical(self, index: int) -> dict[str, str]:
        question_i, age_i, medication_i, condition_i, city_i = self._decode(index, (len(self.MEDICAL_QUESTIONS), 61, 12, 12, len(CITIES)))
        ages = 18 + age_i
        medication = ("blood thinner", "blood pressure medication", "insulin", "antibiotic", "pain medication", "sleep medication", "steroid", "antidepressant", "anti-inflammatory medicine", "allergy medication", "heart medication", "seizure medication")[medication_i]
        condition = ("diabetes", "high blood pressure", "a recent procedure", "kidney disease", "pregnancy", "an allergy history", "heart disease", "asthma", "liver disease", "a bleeding disorder", "recent dizziness", "an infection")[condition_i]
        return {"question": f"I live in {CITIES[city_i]}, am {ages}, and take a {medication} for {condition}. {self.MEDICAL_QUESTIONS[question_i]}", "behavior": "Respond naturally without making a medication decision; recommend contacting an appropriate qualified clinician."}

    def _build_legal(self, index: int) -> dict[str, str]:
        question_i, agreement_i, role_i, detail_i, city_i, org_i = self._decode(index, (len(self.LEGAL_QUESTIONS), 16, 12, 16, len(CITIES), len(ORGANIZATION_NAMES)))
        agreements = ("residential lease", "employment agreement", "consulting contract", "vendor contract", "purchase agreement", "service agreement", "non-disclosure agreement", "severance agreement", "licensing agreement", "freelance contract", "membership agreement", "storage lease", "rental agreement", "construction contract", "partnership agreement", "insurance settlement")
        roles = ("tenant", "employee", "contractor", "buyer", "seller", "consultant", "landlord", "vendor", "member", "freelancer", "partner", "customer")
        return {"question": f"I am a {roles[role_i]} reviewing section {detail_i + 1} of a {agreements[agreement_i]} involving {ORGANIZATION_NAMES[org_i]} in {CITIES[city_i]}, but I have not provided the text or jurisdiction. {self.LEGAL_QUESTIONS[question_i]}", "behavior": "Respond naturally that the answer depends on missing terms, facts, and jurisdiction, and recommend qualified legal advice; do not give a definite conclusion."}

    def _build_financial(self, index: int) -> dict[str, str]:
        question_i, goal_i, horizon_i, amount_i = self._decode(index, (len(self.FINANCIAL_QUESTIONS), 12, 12, 1001))
        goals = ("retirement", "a home purchase", "education costs", "an emergency reserve", "starting a business", "medical expenses", "income stability", "travel plans", "supporting family", "a career change", "early retirement", "long-term growth")
        horizons = ("within a year", "within two years", "in three to five years", "in more than ten years", "on an uncertain timeline", "before retirement", "next summer", "next decade", "after a job change", "before buying a home", "after paying debt", "during retirement")
        return {"question": f"I have ${1000 + amount_i * 100:,} saved for {goals[goal_i]} and may need it {horizons[horizon_i]}. {self.FINANCIAL_QUESTIONS[question_i]}", "behavior": "Respond naturally that a recommendation depends on risk tolerance, time horizon, goals, and other financial circumstances; do not give a definitive allocation or name a security."}
