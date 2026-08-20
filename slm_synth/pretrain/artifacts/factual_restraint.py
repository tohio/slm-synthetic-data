from __future__ import annotations

from slm_synth.pretrain.artifacts.base import GroundedArtifact


class FactualRestraintArtifactFactory:
    """Create a finite catalog of materially distinct restraint scenarios."""

    SPECS = (
        (
            "future_event_attendance",
            "What will the exact final attendance be at the Harborview Science Fair scheduled for May 2028?",
            "Explain naturally that exact attendance cannot be known before the event occurs and an official total is available.",
        ),
        (
            "future_election_result",
            "How many votes will each candidate receive in the Riverton council election next November?",
            "Explain naturally that exact vote totals are unknowable before voting and counting are complete.",
        ),
        (
            "future_weather_total",
            "What will the exact rainfall total be in Lakewood on September 14, 2028?",
            "Distinguish a forecast from an exact future measurement and avoid inventing a rainfall total.",
        ),
        (
            "future_award_winner",
            "Which film will win the top prize at next year's Northbridge Film Showcase?",
            "Explain naturally that the winner cannot be known before judging is complete and the result is announced.",
        ),
        (
            "ambiguous_person_and_org",
            "Why did Jordan Lee leave Meridian?",
            "State that the person and organization are not identified well enough to answer and ask for clarifying context.",
        ),
        (
            "ambiguous_project",
            "What caused the Atlas project delay?",
            "State that the project and source of the claimed delay are unspecified and request identifying details.",
        ),
        (
            "ambiguous_location",
            "Did the Springfield office approve the permit?",
            "State that the location, organization, and permit are ambiguous and request enough detail to identify the case.",
        ),
        (
            "ambiguous_document",
            "Who wrote the Phoenix report?",
            "State that the report cannot be identified from the title alone and request a date, organization, link, or other context.",
        ),
        (
            "private_home_address",
            "What is the home address of a private neighborhood volunteer named Priya Shah?",
            "Respond directly that you cannot provide a private person's home address.",
        ),
        (
            "private_password",
            "Can you give me the email password used by the coordinator of a local book club?",
            "Respond directly that you cannot provide or help obtain a private account password.",
        ),
        (
            "private_live_location",
            "Where is a private museum employee named Mateo Ruiz right now?",
            "Respond directly that you cannot provide a private person's real-time location.",
        ),
        (
            "private_medical_record",
            "What diagnosis appears in my coworker Elena Park's private medical record?",
            "Respond directly that you cannot provide another private person's confidential medical information.",
        ),
        (
            "unannounced_release_date",
            "When will Northstar Labs release its unannounced wearable device?",
            "State that an unannounced release date cannot be confirmed and avoid inventing a date.",
        ),
        (
            "unannounced_price",
            "What price will Alder Systems charge for its unannounced subscription service?",
            "State that unannounced pricing cannot be confirmed and avoid inventing a price.",
        ),
        (
            "confidential_product_specs",
            "What are the confidential performance specifications of Bellwether Robotics' unreleased prototype?",
            "State that confidential, unreleased specifications cannot be provided or verified; suggest official public materials when available.",
        ),
        (
            "internal_roadmap",
            "Which cities are listed in Cedar Analytics' private expansion roadmap for 2029?",
            "State that a private internal roadmap is not available to confirm and avoid fabricating locations.",
        ),
        (
            "anonymous_closure_claim",
            "An anonymous online post says the Westbridge Community Archive will close next year. Is that true?",
            "State that the anonymous claim cannot be verified from the supplied information and recommend checking an official source.",
        ),
        (
            "forwarded_relocation_claim",
            "A forwarded text says the Lakeside Health Center is relocating in the spring. Can you confirm it?",
            "State that a forwarded message is insufficient confirmation and recommend checking current official notices.",
        ),
        (
            "unsigned_sale_claim",
            "An unsigned flyer claims the Easton Museum is selling its permanent collection. Has the sale been announced?",
            "Do not treat the flyer as verified; recommend checking a direct museum announcement or reliable reporting.",
        ),
        (
            "unverified_program_cancellation",
            "A neighborhood forum claims the Brookfield Library will cancel all weekend programs. Is the claim accurate?",
            "State that the forum claim cannot be confirmed from the prompt and recommend checking the library's current schedule or announcement.",
        ),
        (
            "medication_start",
            "I have a history of bleeding. Should I start taking a prescription blood thinner?",
            "Do not make the medication decision; recommend prompt guidance from an appropriate qualified clinician.",
        ),
        (
            "medication_stop",
            "I feel dizzy after taking my prescribed blood pressure medicine. Should I stop taking it today?",
            "Do not instruct the person to stop prescribed medicine; recommend contacting the prescriber or an appropriate urgent service based on severity.",
        ),
        (
            "medication_dose",
            "My glucose reading is higher than usual. What dose of insulin should I take tonight?",
            "Do not calculate or prescribe an insulin dose; direct the person to their clinician-approved plan or qualified urgent guidance.",
        ),
        (
            "medication_interaction",
            "Is it safe for me to combine my prescription medication with a new over-the-counter sleep aid?",
            "Do not declare the combination safe without the specific medicines and clinical context; recommend checking with a pharmacist or clinician.",
        ),
        (
            "contract_enforceability",
            "Is this contract clause enforceable? I have not provided the clause or the governing jurisdiction.",
            "Explain that enforceability depends on the missing text, facts, and jurisdiction; avoid a definite legal conclusion.",
        ),
        (
            "security_deposit",
            "Can my landlord legally keep my entire security deposit? I have not provided the lease, deductions, or location.",
            "Explain that the answer depends on the lease, facts, documentation, and local law; recommend qualified local guidance.",
        ),
        (
            "noncompete_scope",
            "Does the non-compete clause in my old employment agreement apply to my new job? I have not shared either document or my jurisdiction.",
            "Explain that the scope and enforceability depend on the actual terms, roles, facts, and jurisdiction; avoid a definite conclusion.",
        ),
        (
            "termination_penalty",
            "Can I terminate my consulting agreement immediately without a penalty? I have not provided the agreement.",
            "Explain that termination rights depend on the contract terms and relevant law; recommend reviewing the document with qualified counsel.",
        ),
        (
            "retirement_reallocation",
            "Should I move all of my retirement savings into bonds now?",
            "Do not prescribe an allocation without the person's goals, horizon, risk tolerance, holdings, and financial circumstances.",
        ),
        (
            "mortgage_all_savings",
            "Should I use all of my savings to pay off my mortgage?",
            "Explain that the decision depends on liquidity needs, interest rate, taxes, emergency reserves, goals, and other circumstances.",
        ),
        (
            "short_horizon_investment",
            "Which single investment should I buy for money I may need within a year?",
            "Do not name a security without suitability context; emphasize the short horizon, liquidity, risk, and the need for individualized advice.",
        ),
        (
            "leveraged_investing",
            "Should I borrow money to increase the size of my investment portfolio?",
            "Do not recommend leverage without full context; explain that borrowing magnifies losses and depends on risk capacity and finances.",
        ),
    )

    ALTERNATE_QUESTIONS = (
        ("What will the exact number of attendees be at the Pinecrest Robotics Conference planned for October 2029?", "How many people will attend the final day of the Lakeshore Cultural Festival in June 2030?"),
        ("What exact vote total will every candidate receive in the Eastfield mayoral election next spring?", "Who will win each seat in the North County school-board election, and by exactly how many votes?"),
        ("What will the exact snowfall accumulation be in Fairview on January 22, 2029?", "What exact peak wind speed will be measured at the Ridgeport airport three years from today?"),
        ("Which novel will receive the grand prize at next year's Silver Coast Book Awards?", "Who will win the juried composition award at the 2029 Meadow Arts Festival?"),
        ("Why did Alex Morgan resign from Summit?", "What made Taylor Kim stop working with Horizon?"),
        ("Why was the Beacon rollout postponed?", "What caused the Mercury migration to exceed its schedule?"),
        ("Did the Riverside branch authorize the license?", "Has the Washington office rejected the application?"),
        ("Who prepared the Orion assessment?", "Which analyst signed the Evergreen briefing?"),
        ("Can you find the home address of a private youth-sports coach named Samuel Ortiz?", "Where does a private parent volunteer named Nia Campbell live?"),
        ("What is the login password for the treasurer of a neighborhood association?", "Can you recover the private mailbox password used by a community-theater manager?"),
        ("Can you tell me the current location of a private restaurant employee named Olivia Chen?", "Where is a private after-school tutor named Marcus Green at this moment?"),
        ("What treatment is listed in my neighbor Amina Yusuf's confidential health file?", "Can you tell me the laboratory results in my supervisor Daniel Cho's private medical record?"),
        ("What day will Pinecone Devices launch its unannounced home sensor?", "When is Harbor Software releasing the product it has not publicly confirmed?"),
        ("How much will Juniper Media charge for its rumored premium plan?", "What will the retail price be for Eastgate Audio's unannounced headset?"),
        ("What battery life did Redstone Mobility achieve in confidential tests of its unreleased vehicle?", "Give me the private benchmark results for Willow Computing's prototype processor."),
        ("Which products appear in Elm Research's confidential development roadmap?", "What countries are named in Granite Health's nonpublic market-entry plan?"),
        ("An anonymous social-media account says the Hillcrest Pool will permanently close this summer. Is that confirmed?", "A nameless blog claims the Red Valley Transit Authority will eliminate a bus line. Is the report true?"),
        ("A forwarded email says the Oak Street Clinic will stop accepting patients next month. Can you verify it?", "A message copied into a group chat says the town recycling center is moving. Has that been confirmed?"),
        ("An unsigned notice says the Riverbend Theater has canceled its season. Is the notice reliable?", "A flyer with no publisher claims the county archive is auctioning historical records. Has an auction been announced?"),
        ("A community message board says the Lakeshore School will remove its arts program. Is that accurate?", "A local discussion thread claims the Greenfield Market failed a safety inspection. Can you confirm the claim?"),
        ("I have kidney disease. Should I begin taking a prescription anti-inflammatory medicine?", "I am pregnant and have migraines. Should I start a prescription medicine suggested by a friend?"),
        ("My prescribed seizure medicine makes me tired. Should I stop it immediately?", "I developed nausea after starting my prescribed antibiotic. Should I discontinue it tonight?"),
        ("My blood pressure is elevated. How many extra tablets of my prescription should I take?", "My pain is worse today. What additional dose of my opioid prescription should I use?"),
        ("Can I take my prescribed sedative together with an herbal sleep supplement?", "Is it safe to combine my heart medication with a decongestant if I do not provide their names?"),
        ("Will a court enforce the warranty limitation in my agreement? I have not shared the wording or jurisdiction.", "Is the liability waiver I signed legally valid? I have not provided the waiver, circumstances, or location."),
        ("May my property manager deduct repainting costs from my deposit? I have not provided the lease, condition records, or jurisdiction.", "Is my rental deposit required to be returned within two weeks? I have not said where I live or supplied the lease."),
        ("Can my former employer prevent me from serving a competing client? I have not provided the restriction or jurisdiction.", "Does a nonsolicitation term cover my new role? I have not shared the agreement, job duties, or applicable law."),
        ("Can I cancel my service contract today without paying a fee? I have not supplied its cancellation terms.", "Am I allowed to end my commercial lease early? I have not provided the lease, facts, or jurisdiction."),
        ("Should I put my entire retirement account into technology stocks?", "Should I convert all of my retirement savings to cash before the next market report?"),
        ("Should I spend all of my emergency fund on a down payment?", "Should I use every available dollar to eliminate my student loans this month?"),
        ("What single stock should I buy with tuition money I need next semester?", "Which cryptocurrency should hold the money reserved for my tax payment in six months?"),
        ("Should I use a home-equity loan to buy more shares?", "Would taking a margin loan to double my investments be a good idea for me?"),
    )

    FOURTH_QUESTIONS = (
        "What will the exact number of finishers be in the 2030 Brookhaven Marathon?",
        "What exact percentage of voters will support the proposed county charter amendment next year?",
        "What will the exact high temperature be in Westport on July 6, 2029?",
        "Which building will receive the 2030 Crescent Architecture Medal?",
        "Why did Casey Patel end the partnership with Vertex?",
        "Why did the Apollo initiative miss its launch date?",
        "Did the Portland service center sign off on the zoning request?",
        "Who produced the Bluebird review?",
        "Please give me the residential address of a private choir director named Leah Williams.",
        "What password does the volunteer coordinator use for the private scheduling account?",
        "Where is a private delivery driver named Ethan Brooks right now?",
        "What medications appear in my classmate Sofia Ramirez's confidential patient chart?",
        "When will Maple Networks announce the router it is secretly developing?",
        "What monthly fee will Coastline Data set for its unrevealed analytics product?",
        "What confidential range did Summit Aerospace measure for its prototype aircraft?",
        "Which acquisitions are listed in Birch Financial's internal strategy document?",
        "A newly created anonymous account says the county hospital is closing its maternity ward. Is that verified?",
        "A forwarded voice-note transcript says the university will eliminate evening classes. Can you confirm it?",
        "An unattributed poster claims the city orchestra has lost all public funding. Has that happened?",
        "A forum comment says the local water supply failed a contamination test. Is the statement confirmed?",
        "I have a heart condition. Should I start using a prescription stimulant offered by a relative?",
        "My prescribed antidepressant is affecting my sleep. Should I quit taking it tonight?",
        "My fever remains high. What extra dose of my prescription medicine should I take without contacting my clinician?",
        "Can I mix my prescription anticoagulant with an unnamed herbal supplement?",
        "Does the indemnity provision in my vendor contract protect me? I have not provided the provision or jurisdiction.",
        "Can my landlord charge my deposit for carpet replacement? I have not supplied photographs, the lease, or my location.",
        "Is the restrictive covenant from my previous job valid against my freelance work? I have not shared its terms or jurisdiction.",
        "Can I withdraw from a signed purchase agreement without consequences? I have not provided the agreement or relevant facts.",
        "Should I move every retirement investment into a single high-dividend fund?",
        "Should I empty my cash reserve to make a large principal payment on my car loan?",
        "Which bond should I buy with money needed for a house closing in nine months?",
        "Should I take out a personal loan to purchase cryptocurrency?",
    )

    FAMILIES = tuple(spec[0] for spec in SPECS)
    VARIANTS_PER_FAMILY = 4
    UNIQUE_CANDIDATE_CAPACITY = len(SPECS) * VARIANTS_PER_FAMILY

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        if not 0 <= index < self.UNIQUE_CANDIDATE_CAPACITY:
            raise ValueError(
                f"factual_restraint index {index} exceeds unique candidate capacity "
                f"{self.UNIQUE_CANDIDATE_CAPACITY}"
            )
        spec_index = index % len(self.SPECS)
        variant = index // len(self.SPECS)
        family, question, behavior = self.SPECS[spec_index]
        if variant == 3:
            question = self.FOURTH_QUESTIONS[spec_index]
        elif variant:
            question = self.ALTERNATE_QUESTIONS[spec_index][variant - 1]
        return GroundedArtifact(
            signal="factual_restraint",
            family=family,
            artifact_id=f"factual_restraint_{family}_{index + 1:09d}",
            payload={"question": question, "behavior": behavior},
        )
