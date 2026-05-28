from __future__ import annotations

import random

from slm_synth.artifacts.base import GroundedArtifact
from slm_synth.artifacts.lexicon import CITIES, FIRST_NAMES, LAST_NAMES, ORGANIZATION_NAMES, VENUES, full_name


class EducationalQAMCQGeneralArtifactFactory:
    """Create deterministic, natural evidence-grounded MCQs."""

    FAMILIES = (
        "python_behavior", "grammar", "vocabulary", "reading", "fictional_rule", "policy", "scientific_method", "ordering",
        "final_location", "table_lookup", "threshold_rule", "temporal_order", "direction_following", "conditional_access",
        "comparison_claim", "category_rule", "cause_inference", "schedule_availability", "inventory_shortage", "source_attribution",
        "procedure_step", "exception_rule", "trend_interpretation", "revision_tracking",
    )
    PLACES = ("blue drawer", "green cabinet", "front desk", "storage shelf", "toolbox", "library cart", "archive bin", "side locker", "supply room", "service counter", "upper cabinet", "lower tray")
    OBJECTS = ("spare key", "signed forms", "visitor badge", "survey folder", "repair manual", "supply list", "delivery slip", "permit copy", "tool record", "inventory sheet", "registration card", "inspection note")
    ADVERBS = ("quietly", "carefully", "quickly", "patiently", "neatly", "calmly", "slowly", "briefly", "softly", "promptly")
    ADJECTIVE_CONTEXT = {
        "fragile": ("easily broken", "It needed protective wrapping."),
        "narrow": ("not wide", "Only one person could pass through at a time."),
        "ancient": ("very old", "It had been preserved for centuries."),
        "bright": ("giving much light", "It illuminated the nearby shelf."),
        "careful": ("taking caution", "No details were overlooked."),
        "silent": ("not making sound", "No sound could be heard there."),
        "scarce": ("hard to find", "Only a few remained in stock."),
        "durable": ("long-lasting", "It survived repeated daily use."),
        "precise": ("exact", "Every measurement matched the specification."),
        "modest": ("not excessive", "It required only a small amount."),
    }
    # Use adjective-compatible subjects instead of pairing every clue with the
    # generic OBJECTS list. Each adjective retains 12 distinct subject forms so
    # vocabulary artifact diversity is not reduced by this coherence fix.
    VOCABULARY_SUBJECTS = {
        "fragile": ("glass ornament", "ceramic sample", "thin vial", "display model", "porcelain tile", "glass slide", "clay figurine", "crystal cup", "sample tube", "decorative plate", "glass panel", "ceramic bowl"),
        "narrow": ("storage aisle", "service doorway", "passageway", "hallway entrance", "garden gate", "archive corridor", "loading ramp", "footbridge", "stairwell", "service passage", "supply aisle", "side entrance"),
        "ancient": ("manuscript", "stone tablet", "map", "ceremonial bowl", "bronze coin", "woven tapestry", "clay tablet", "wooden carving", "archive scroll", "mosaic tile", "inscribed seal", "painted vase"),
        "bright": ("inspection lamp", "desk lamp", "display light", "work light", "ceiling fixture", "reading lamp", "portable lantern", "light panel", "task lamp", "spotlight", "window display", "emergency beacon"),
        "careful": ("records inspector", "lab technician", "reviewer", "inventory clerk", "archive assistant", "quality auditor", "field researcher", "equipment handler", "safety officer", "proofreader", "curator", "surveyor"),
        "silent": ("reading room", "hallway", "workroom", "archive room", "empty theater", "study area", "closed gallery", "waiting room", "conference room", "library wing", "office corridor", "practice studio"),
        "scarce": ("replacement battery", "repair part", "shipping label", "filter cartridge", "safety mask", "sample jar", "packing box", "tool insert", "printer ribbon", "storage clip", "seal kit", "cleaning pad"),
        "durable": ("tool case", "storage bin", "work glove", "canvas bag", "safety helmet", "travel crate", "utility belt", "floor mat", "metal cart", "protective cover", "field notebook", "equipment strap"),
        "precise": ("measuring instrument", "digital scale", "calibration gauge", "timing device", "laser ruler", "temperature probe", "pressure meter", "survey level", "micrometer", "volume dispenser", "alignment tool", "laboratory balance"),
        "modest": ("supply request", "budget request", "equipment order", "space request", "travel allowance", "printing order", "staffing request", "repair estimate", "meal budget", "storage request", "material order", "grant request"),
    }
    VERBS = ("stored", "placed", "carried", "moved", "checked", "returned", "sealed", "organized", "recorded", "reviewed")
    FICTIONAL_REGIONS = ("Orin", "Pelin", "Nareth", "Volar", "Kesra", "Tavon", "Merin", "Solara", "Bren", "Doria", "Calen", "Ivara", "Aster", "Belin", "Corin", "Elora", "Faron", "Galen", "Haven", "Joren")
    FICTIONAL_LABELS = ("Velora", "Moro", "Sorin", "Kessa", "Tavin", "Liora", "Brelan", "Norvi", "Selka", "Dorin", "Pavri", "Merola", "Quorin", "Vessa", "Talune", "Rovik")
    MARK_COLORS = ("silver", "blue", "red", "amber", "green", "white", "black", "copper")
    MARK_SHAPES = ("handle", "stripe", "circle", "flag", "seal", "ribbon", "triangle", "star")
    DEPARTMENTS = ("records", "finance", "research", "facilities", "archive", "security", "clinical", "procurement")
    VARIABLES = ("type of music", "amount of light", "water temperature", "type of soil", "room temperature", "amount of water", "container size", "time exposed")
    DIRECTIONS = ("north", "east", "south", "west")
    TIME_SLOTS = ("08:00", "09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00")
    CATEGORIES = ("routine", "priority", "restricted", "archival", "fragile", "urgent", "review", "approved")
    PROCEDURE_STEPS = ("inspect the seal", "record the identifier", "photograph the label", "store the item")
    TREND_LABELS = ("increased", "decreased", "stayed constant", "fluctuated")

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
        self._shuffle_choices(payload, index)
        return GroundedArtifact("educational_qa_mcq_general", family, f"educational_qa_mcq_general_{family}_{index + 1:09d}", payload)

    @staticmethod
    def _shuffle_choices(payload: dict[str, object], index: int) -> None:
        """Deterministically randomize choice positions without changing the answer."""
        choices = list(payload["choices"])
        answer = str(payload["answer"])
        random.Random(f"educational_qa_mcq_general:{index}").shuffle(choices)
        payload["choices"] = choices
        payload["correct_index"] = choices.index(answer)

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
        name, place = full_name(person_i), self.PLACES[place_i]
        subject = self.VOCABULARY_SUBJECTS[adjective][obj_i]
        evidence = f'Sentence: "In a note stored in the {place}, {name} described the {subject} as {adjective}. {clue}"'
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


    def _build_final_location(self, index: int) -> dict[str, object]:
        place_count = len(self.PLACES)
        ordered_route_count = place_count * (place_count - 1) * (place_count - 2)
        obj_i, route_i, person_i = self._decode(
            index,
            (len(self.OBJECTS), ordered_route_count, len(FIRST_NAMES) * len(LAST_NAMES)),
        )
        obj, name = self.OBJECTS[obj_i], full_name(person_i)

        first_slot = route_i % place_count
        route_i //= place_count
        second_slot = route_i % (place_count - 1)
        route_i //= place_count - 1
        third_slot = route_i % (place_count - 2)

        remaining = list(self.PLACES)
        first_place = remaining.pop(first_slot)
        second_place = remaining.pop(second_slot)
        final_place = remaining[third_slot]
        places = [first_place, second_place, final_place]

        evidence = f"Movement log: {name} first put the {obj} in the {places[0]}, then moved it to the {places[1]}, and finally transferred it to the {final_place}."
        alternatives = [place for place in self.PLACES if place != final_place][:3]
        choices = [f"the {final_place}"] + [f"the {place}" for place in alternatives]
        return self._record(evidence, f"After moving the {obj} along the route {places[0]} -> {places[1]} -> {final_place}, where does {name} leave it?", choices, f"the {final_place}")

    def _build_table_lookup(self, index: int) -> dict[str, object]:
        dept_i, offset_i, city_i, person_i = self._decode(index, (len(self.DEPARTMENTS), 17, len(CITIES), len(FIRST_NAMES) * len(LAST_NAMES)))
        departments = [self.DEPARTMENTS[(dept_i + step) % len(self.DEPARTMENTS)] for step in range(4)]
        values = [18 + offset_i, 31 + offset_i, 24 + offset_i, 12 + offset_i]
        analyst = full_name(person_i)
        evidence = f"Report table prepared by {analyst} for {CITIES[city_i]}: " + "; ".join(f"{dept}={value} requests" for dept, value in zip(departments, values)) + "."
        answer = departments[1]
        return self._record(evidence, f"In {analyst}'s {CITIES[city_i]} report with counts {values}, which department recorded the most requests?", departments, answer)

    def _build_threshold_rule(self, index: int) -> dict[str, object]:
        dept_i, item_i, offset_i, person_i = self._decode(index, (len(self.DEPARTMENTS), len(self.OBJECTS), 11, len(FIRST_NAMES) * len(LAST_NAMES)))
        dept, item, reviewer = self.DEPARTMENTS[dept_i], self.OBJECTS[item_i], full_name(person_i)
        threshold = 20 + offset_i
        counts = [threshold - 3, threshold + 2, threshold - 1, threshold - 5]
        labels = [f"Batch {label}: {count} {item}" for label, count in zip(("A", "B", "C", "D"), counts)]
        evidence = f"Review assignment: {reviewer} on the {dept} team escalates a batch of {item} only when its count exceeds {threshold}. " + "; ".join(labels) + "."
        return self._record(evidence, f"Which {item} batch must {reviewer} escalate for the {dept} team?", labels, labels[1])

    def _build_temporal_order(self, index: int) -> dict[str, object]:
        obj_i, start_i, place_i, person_i = self._decode(index, (len(self.OBJECTS), len(self.TIME_SLOTS) - 3, len(self.PLACES), len(FIRST_NAMES) * len(LAST_NAMES)))
        name, obj, place = full_name(person_i), self.OBJECTS[obj_i], self.PLACES[place_i]
        events = ["checked", "sealed", "recorded", "stored"]
        times = self.TIME_SLOTS[start_i:start_i + 4]
        evidence = f"Timeline for the {obj} at the {place}: " + "; ".join(f"at {time}, {name} {event} it" for time, event in zip(times, events)) + "."
        choices = [f"{event} the {obj}" for event in events]
        return self._record(evidence, f"At the {place} in the timeline beginning at {times[0]}, what did {name} do immediately before storing the {obj}?", choices, f"recorded the {obj}")

    def _build_direction_following(self, index: int) -> dict[str, object]:
        obj_i, direction_i, place_i, person_i = self._decode(index, (len(self.OBJECTS), len(self.DIRECTIONS), len(self.PLACES), len(FIRST_NAMES) * len(LAST_NAMES)))
        obj, direction, guide = self.OBJECTS[obj_i], self.DIRECTIONS[direction_i], full_name(person_i)
        opposite = self.DIRECTIONS[(direction_i + 2) % 4]
        evidence = f"Map note from {guide}: From the {self.PLACES[place_i]}, walk one block {direction} and place the {obj} there. A separate marker lies one block {opposite}."
        return self._record(evidence, f"Following {guide}'s map note, in which direction is the {obj} from the {self.PLACES[place_i]}?", list(self.DIRECTIONS), direction)

    def _build_conditional_access(self, index: int) -> dict[str, object]:
        dept_i, obj_i, place_i, person_i = self._decode(index, (len(self.DEPARTMENTS), len(self.OBJECTS), len(self.PLACES), len(FIRST_NAMES) * len(LAST_NAMES)))
        dept, obj, place, supervisor = self.DEPARTMENTS[dept_i], self.OBJECTS[obj_i], self.PLACES[place_i], full_name(person_i)
        evidence = f"Access rule approved by {supervisor}: a person may remove the {obj} from the {place} only if they are on the {dept} team and carry a signed pass."
        allowed = f"A {dept} employee with a signed pass removes the {obj}."
        choices = [allowed, f"A {dept} employee without a pass removes the {obj}.", f"A visitor with a signed pass removes the {obj}.", f"A visitor without a pass removes the {obj}."]
        return self._record(evidence, f"Under {supervisor}'s access rule, which action is permitted for the {obj} kept in the {place}?", choices, allowed)

    def _build_comparison_claim(self, index: int) -> dict[str, object]:
        item_i, offset_i, city_i, person_i = self._decode(index, (len(self.OBJECTS), 23, len(CITIES), len(FIRST_NAMES) * len(LAST_NAMES)))
        item, analyst = self.OBJECTS[item_i], full_name(person_i)
        values = [12 + offset_i, 19 + offset_i, 15 + offset_i, 9 + offset_i]
        labels = ["North", "East", "South", "West"]
        evidence = f"Count summary prepared by {analyst} for {item} in {CITIES[city_i]}: " + "; ".join(f"{label} office={value}" for label, value in zip(labels, values)) + "."
        claims = [f"The {label} office has {value} {item}." for label, value in zip(labels, values)]
        return self._record(evidence, f"In {analyst}'s {CITIES[city_i]} summary, which office has the largest reported count of {item}?", claims, claims[1])

    def _build_category_rule(self, index: int) -> dict[str, object]:
        category_i, obj_i, color_i, person_i = self._decode(index, (len(self.CATEGORIES), len(self.OBJECTS), len(self.MARK_COLORS), len(FIRST_NAMES) * len(LAST_NAMES)))
        category, obj, color, clerk = self.CATEGORIES[category_i], self.OBJECTS[obj_i], self.MARK_COLORS[color_i], full_name(person_i)
        evidence = f"Sorting guide used by {clerk}: Any package bearing a {color} seal is filed as {category}. The package containing the {obj} bears a {color} seal."
        choices = [category] + [self.CATEGORIES[(category_i + step) % len(self.CATEGORIES)] for step in (1, 2, 3)]
        return self._record(evidence, f"Using {clerk}'s guide, how should the {color}-sealed package containing the {obj} be filed?", choices, category)

    def _build_cause_inference(self, index: int) -> dict[str, object]:
        obj_i, condition_i, place_i, person_i = self._decode(index, (len(self.OBJECTS), 4, len(self.PLACES), len(FIRST_NAMES) * len(LAST_NAMES)))
        obj, place, observer = self.OBJECTS[obj_i], self.PLACES[place_i], full_name(person_i)
        conditions = ("water was added", "the lamp was switched on", "the heater was turned on", "the cover was removed")
        effects = ("the container became heavier", "the work surface became brighter", "the reading increased in temperature", "the item became visible")
        cause, effect = conditions[condition_i], effects[condition_i]
        evidence = f"Observation by {observer} at the {place}: Before {cause}, no change was recorded for the {obj}. Immediately after {cause}, {effect}. No other action occurred."
        choices = [cause] + [conditions[(condition_i + step) % len(conditions)] for step in (1, 2, 3)]
        return self._record(evidence, f"According to {observer}'s observation, which action most directly explains the change in the {obj} at the {place}?", choices, cause)

    def _build_schedule_availability(self, index: int) -> dict[str, object]:
        venue_i, start_i, person_i = self._decode(index, (len(VENUES), len(self.TIME_SLOTS) - 3, len(FIRST_NAMES) * len(LAST_NAMES)))
        venue, name = VENUES[venue_i], full_name(person_i)
        times = self.TIME_SLOTS[start_i:start_i + 4]
        evidence = f"Booking sheet for the {venue}: {times[0]} booked, {times[1]} booked, {times[2]} available, {times[3]} booked. {name} needs an available slot."
        return self._record(evidence, f"Which time can {name} reserve at the {venue} among {', '.join(times)}?", list(times), times[2])

    def _build_inventory_shortage(self, index: int) -> dict[str, object]:
        item_i, offset_i, dept_i, person_i = self._decode(index, (len(self.OBJECTS), 13, len(self.DEPARTMENTS), len(FIRST_NAMES) * len(LAST_NAMES)))
        item, dept, planner = self.OBJECTS[item_i], self.DEPARTMENTS[dept_i], full_name(person_i)
        required = 30 + offset_i
        quantities = [required + 5, required - 4, required + 1, required + 8]
        locations = [self.PLACES[(item_i + step) % len(self.PLACES)] for step in range(4)]
        evidence = f"Inventory review by {planner}: The {dept} team requires at least {required} copies of the {item}. " + "; ".join(f"{place} holds {quantity}" for place, quantity in zip(locations, quantities)) + "."
        return self._record(evidence, f"According to {planner}'s review, which location cannot supply the required {required} {item} for the {dept} team?", locations, locations[1])

    def _build_source_attribution(self, index: int) -> dict[str, object]:
        first_i, second_i, obj_i, place_i = self._decode(index, (len(FIRST_NAMES) * len(LAST_NAMES), len(FIRST_NAMES) * len(LAST_NAMES), len(self.OBJECTS), len(self.PLACES)))
        first, second = full_name(first_i), full_name((second_i + 1) % (len(FIRST_NAMES) * len(LAST_NAMES)))
        if first == second:
            second = full_name((second_i + 2) % (len(FIRST_NAMES) * len(LAST_NAMES)))
        obj, place = self.OBJECTS[obj_i], self.PLACES[place_i]
        evidence = f"Message log: {first} submitted the {obj} for review. {second} later stored the reviewed item in the {place}."
        choices = [first, second, "the reviewer", "the storage clerk"]
        return self._record(evidence, f"Who submitted the {obj} later stored in the {place} for review?", choices, first)

    def _build_procedure_step(self, index: int) -> dict[str, object]:
        obj_i, place_i, shift_i, person_i = self._decode(index, (len(self.OBJECTS), len(self.PLACES), len(self.PROCEDURE_STEPS), len(FIRST_NAMES) * len(LAST_NAMES)))
        obj, place, trainer = self.OBJECTS[obj_i], self.PLACES[place_i], full_name(person_i)
        steps = list(self.PROCEDURE_STEPS[shift_i:]) + list(self.PROCEDURE_STEPS[:shift_i])
        evidence = f"Procedure assigned by {trainer} for the {obj} at the {place}: first {steps[0]}, next {steps[1]}, then {steps[2]}, and finally {steps[3]}."
        choices = steps
        return self._record(evidence, f"In {trainer}'s procedure, which step occurs immediately after the first step for the {obj} at the {place}?", choices, steps[1])

    def _build_exception_rule(self, index: int) -> dict[str, object]:
        dept_i, obj_i, place_i, person_i = self._decode(index, (len(self.DEPARTMENTS), len(self.OBJECTS), len(self.PLACES), len(FIRST_NAMES) * len(LAST_NAMES)))
        dept, obj, place, manager = self.DEPARTMENTS[dept_i], self.OBJECTS[obj_i], self.PLACES[place_i], full_name(person_i)
        evidence = f"Handling rule issued by {manager}: all {obj} must remain in the {place}, except when a {dept} supervisor authorizes temporary inspection."
        exception = f"A {dept} supervisor authorizes temporary inspection of the {obj}."
        choices = [exception, f"A visitor removes the {obj} without approval.", f"A clerk discards the {obj}.", f"A courier takes the {obj} permanently."]
        return self._record(evidence, f"Under {manager}'s rule, which situation is an allowed exception for the {obj} stored in the {place}?", choices, exception)

    def _build_trend_interpretation(self, index: int) -> dict[str, object]:
        item_i, pattern_i, offset_i, place_i, person_i = self._decode(index, (len(self.OBJECTS), len(self.TREND_LABELS), 10, len(self.PLACES), len(FIRST_NAMES) * len(LAST_NAMES)))
        item, label, place, analyst = self.OBJECTS[item_i], self.TREND_LABELS[pattern_i], self.PLACES[place_i], full_name(person_i)
        start = 10 + offset_i
        series = {
            "increased": [start, start + 2, start + 4, start + 6],
            "decreased": [start + 6, start + 4, start + 2, start],
            "stayed constant": [start, start, start, start],
            "fluctuated": [start, start + 3, start + 1, start + 4],
        }[label]
        evidence = f"Weekly counts compiled by {analyst} for the {item} at the {place}: week 1={series[0]}, week 2={series[1]}, week 3={series[2]}, week 4={series[3]}."
        return self._record(evidence, f"Which description best matches {analyst}'s weekly counts {series} for the {item} at the {place}?", list(self.TREND_LABELS), label)

    def _build_revision_tracking(self, index: int) -> dict[str, object]:
        obj_i, place_i, first_i, second_i = self._decode(index, (len(self.OBJECTS), len(self.PLACES), len(FIRST_NAMES) * len(LAST_NAMES), len(FIRST_NAMES) * len(LAST_NAMES)))
        obj, place = self.OBJECTS[obj_i], self.PLACES[place_i]
        first = full_name(first_i)
        second = full_name((second_i + 3) % (len(FIRST_NAMES) * len(LAST_NAMES)))
        if first == second:
            second = full_name((second_i + 4) % (len(FIRST_NAMES) * len(LAST_NAMES)))
        evidence = f"Revision history: Version 1 of the {obj} note was created by {first}. Version 2 corrected the storage location to the {place} and was approved by {second}."
        choices = [first, second, "the records office", "the storage team"]
        return self._record(evidence, f"Who approved the corrected version of the {obj} note stored in the {place}?", choices, second)
