import json

import pytest

from slm_synth.distillation_sft.response_adjudication import (
    apply_response_cluster_adjudications,
)
from slm_synth.distillation_sft.response_diversity import build_response_diversity_summary


def _cloud_rows() -> list[dict[str, object]]:
    repeated_response = (
        "Use a load balancer with autoscaling, health checks, monitoring, "
        "and documented recovery procedures."
    )
    rows = []
    for index in range(29):
        row_id = "cloud-002448" if index == 28 else f"cloud-{index + 1:06d}"
        prompt = (
            "Design a resilient API using a load balancer, autoscaling, health checks, "
            "monitoring, and recovery procedures."
            if row_id == "cloud-002448"
            else f"Explain distinct cloud architecture concern {index}."
        )
        rows.append(
            {
                "id": row_id,
                "prompt": prompt,
                "reasoning": None,
                "response": repeated_response,
                "metadata": {
                    "category": "general_instruction_following",
                    "difficulty": 2,
                    "template_family": f"cloud_audit_case_{index % 5}",
                    "eval_family": None,
                },
            }
        )
    return rows


def _write_rows(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _decisions_for_cluster(cluster) -> dict[str, object]:
    return {
        "schema_version": 1,
        "decisions": [
            {
                "member_fingerprint": member["member_fingerprint"],
                "decision": "keep" if member["id"] == "cloud-002448" else "reject",
                "reason": (
                    "response matches this prompt"
                    if member["id"] == "cloud-002448"
                    else "response does not answer this prompt"
                ),
            }
            for member in cluster["members"]
        ],
    }


def test_adjudication_keeps_valid_control_and_quarantines_other_cluster_members(tmp_path):
    dataset_dir = tmp_path / "datasets"
    rejected_dir = tmp_path / "rejected"
    dataset_dir.mkdir()
    dataset = dataset_dir / "cloud.jsonl"
    _write_rows(dataset, _cloud_rows())
    cluster = build_response_diversity_summary([dataset])["repeated_response_clusters"][0]
    adjudications = tmp_path / "adjudications.json"
    adjudications.write_text(json.dumps(_decisions_for_cluster(cluster)), encoding="utf-8")

    summary = apply_response_cluster_adjudications(
        dataset_dir=dataset_dir,
        adjudications_path=adjudications,
        rejected_dir=rejected_dir,
    )

    kept = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    rejected = [
        json.loads(line)
        for line in (rejected_dir / "repeated_response_adjudications.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert summary["reviewed_rows"] == 29
    assert summary["kept_rows"] == 1
    assert summary["rejected_rows"] == 28
    assert [row["id"] for row in kept] == ["cloud-002448"]
    assert len(rejected) == 28
    assert "cloud-002448" not in {record["row"]["id"] for record in rejected}
    assert all(record["rejection_reason"] == "repeated_response_cluster_adjudication" for record in rejected)


def test_incomplete_adjudication_fails_before_changing_dataset(tmp_path):
    dataset_dir = tmp_path / "datasets"
    rejected_dir = tmp_path / "rejected"
    dataset_dir.mkdir()
    dataset = dataset_dir / "cloud.jsonl"
    _write_rows(dataset, _cloud_rows())
    original = dataset.read_bytes()
    cluster = build_response_diversity_summary([dataset])["repeated_response_clusters"][0]
    payload = _decisions_for_cluster(cluster)
    payload["decisions"] = payload["decisions"][:-1]
    adjudications = tmp_path / "adjudications.json"
    adjudications.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or stale"):
        apply_response_cluster_adjudications(
            dataset_dir=dataset_dir,
            adjudications_path=adjudications,
            rejected_dir=rejected_dir,
        )

    assert dataset.read_bytes() == original
    assert not rejected_dir.exists()
