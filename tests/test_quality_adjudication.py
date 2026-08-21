import copy

import pytest

from slm_synth.dpo.generation import DPOBatchAcceptanceError, generate_llm_batch as generate_dpo_batch
from slm_synth.dpo.spec_builders import build_specs as build_dpo_specs
from slm_synth.model_contract import parse_judge_decision, parse_review_decision
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend, StagedDPOBackend
from tests.test_dpo_llm_generation import Backend as CompleteDPOBackend


def test_judge_rejects_unassessable_even_if_decision_says_accept():
    decision = parse_judge_decision(
        "ASSESSABLE: NO\nDECISION: ACCEPT\nREASON: Required evidence is missing."
    )
    assert decision.assessable is False
    assert decision.accepted is False


def test_plain_decision_contracts_reject_missing_labels():
    with pytest.raises(ValueError, match="missing labeled"):
        parse_judge_decision("DECISION: ACCEPT")
    with pytest.raises(ValueError, match="missing labeled"):
        parse_review_decision("AGREE: YES")


def test_reviewer_disagreement_rejects_without_regeneration(tmp_path):
    class DisagreeingReviewer:
        def generate_text_with_metadata(self, *, prompt, system_prompt):
            return {"text": "AGREE: NO\nREASON: The judge missed unsupported content.", "telemetry": {}}

    result = generate_dpo_batch(
        specs=build_dpo_specs(family="factual_accuracy", count=1),
        output_path=tmp_path / "dpo.jsonl", manifest_path=tmp_path / "manifest.json",
        teacher_model="teacher/model", generation_run="review-reject", max_tokens=1024,
        backend=StagedDPOBackend(CompleteDPOBackend()),
        adjudicator_backend=AcceptingAdjudicatorBackend(), reviewer_backend=DisagreeingReviewer(),
    )
    assert result.row_count == 0
    assert result.semantic_rejected_count == 1
    assert (tmp_path / "dpo.jsonl").read_text() == ""


def test_dpo_identical_branches_are_rejected_without_local_repair(tmp_path):
    spec = build_dpo_specs(family="factual_accuracy", count=1)[0]

    class IdenticalBackend(CompleteDPOBackend):
        def _row(self, value):
            row = super()._row(value)
            row["rejected"] = copy.deepcopy(row["chosen"])
            return row

    with pytest.raises(DPOBatchAcceptanceError, match="must differ"):
        generate_dpo_batch(
            specs=[spec], output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "manifest.json", teacher_model="teacher/model",
            generation_run="no-repair", max_tokens=1024,
            backend=StagedDPOBackend(IdenticalBackend()),
            adjudicator_backend=AcceptingAdjudicatorBackend(),
        )
