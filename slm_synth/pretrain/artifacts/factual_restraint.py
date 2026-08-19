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

    FAMILIES = tuple(spec[0] for spec in SPECS)
    UNIQUE_CANDIDATE_CAPACITY = len(SPECS)

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        if not 0 <= index < self.UNIQUE_CANDIDATE_CAPACITY:
            raise ValueError(
                f"factual_restraint index {index} exceeds unique candidate capacity "
                f"{self.UNIQUE_CANDIDATE_CAPACITY}"
            )
        family, question, behavior = self.SPECS[index]
        return GroundedArtifact(
            signal="factual_restraint",
            family=family,
            artifact_id=f"factual_restraint_{family}_{index + 1:09d}",
            payload={"question": question, "behavior": behavior},
        )
