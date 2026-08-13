import json

from slm_synth.distillation_sft.response_diversity import (
    build_response_diversity_summary,
    normalize_response_text,
)


def _row(row_id: str, *, prompt: str, response: str) -> dict[str, object]:
    return {
        "id": row_id,
        "prompt": prompt,
        "reasoning": None,
        "response": response,
        "metadata": {
            "category": "general_instruction_following",
            "difficulty": 2,
            "template_family": "python_optional_key_bug",
            "eval_family": None,
        },
    }


def _write_rows(path, responses):
    rows = [
        _row(f"{path.stem}-{index}", prompt=f"Prompt {index}", response=response)
        for index, response in enumerate(responses)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_normalize_response_text_ignores_case_and_whitespace():
    assert normalize_response_text("  A\n  concise RESPONSE. ") == "a concise response."


def test_build_response_diversity_summary_reports_each_signal(tmp_path):
    debugging = tmp_path / "debugging.jsonl"
    planning = tmp_path / "planning.jsonl"
    _write_rows(debugging, ["same", "same", "different", "another"])
    _write_rows(planning, ["plan one", "plan two"])

    summary = build_response_diversity_summary([debugging, planning])

    assert summary["row_count"] == 6
    assert summary["unique_response_count"] == 5
    assert summary["duplicate_response_count"] == 1
    assert summary["signals"]["debugging"] == {
        "row_count": 4,
        "unique_response_count": 3,
        "duplicate_response_count": 1,
        "unique_response_ratio": 0.75,
        "duplicate_examples": [{"response": "same", "count": 2}],
    }
    assert summary["signals"]["planning"]["unique_response_ratio"] == 1.0
    cluster = summary["repeated_response_clusters"][0]
    assert cluster["count"] == 2
    assert cluster["normalized_response"] == "same"
    assert cluster["responses"] == ["same"]
    assert [member["id"] for member in cluster["members"]] == ["debugging-0", "debugging-1"]


def test_repeated_response_clusters_expose_all_29_cloud_rows_including_valid_control(tmp_path):
    cloud = tmp_path / "cloud.jsonl"
    repeated_response = (
        "Use a load balancer with autoscaling, health checks, monitoring, and documented recovery procedures."
    )
    valid_control_id = "cloud-002448"
    prompts = [
        "Design a batch-processing system that scales workers with queue depth.",
        "Design a multi-zone service that remains available after one zone fails.",
        "Explain how to isolate development, staging, and production environments.",
        "Design durable object storage for user uploads.",
        "Design an API service that scales horizontally and recovers from unhealthy instances.",
        "Reduce latency for a globally distributed read-heavy application.",
        "Reduce cloud cost for an interruptible batch workload.",
        "Protect a public API using least privilege and layered security controls.",
        "Design a fault-tolerant event-processing pipeline.",
        "Choose a deployment strategy that minimizes release risk.",
        "Design database backups with tested point-in-time recovery.",
        "Plan capacity for a predictable seasonal traffic spike.",
        "Design tenant isolation for a multi-tenant SaaS application.",
        "Choose a caching strategy for frequently requested product data.",
        "Design secure secret storage and rotation for application credentials.",
        "Design centralized logging for services running in several accounts.",
        "Design a disaster-recovery plan with explicit recovery objectives.",
        "Choose a messaging pattern for decoupled order processing.",
        "Design network segmentation between public and private workloads.",
        "Design a data-retention policy for regulated audit records.",
        "Plan a zero-downtime relational database migration.",
        "Design monitoring and alerting for a latency-sensitive service.",
        "Choose storage for temporary files produced by short-lived workers.",
        "Design an access-review process for privileged cloud roles.",
        "Plan regional failover for a customer-facing application.",
        "Design rate limiting for an externally accessible API.",
        "Choose a cost-allocation approach for shared platform services.",
        "Design data encryption controls for stored customer records.",
        "Design a resilient API using a load balancer, autoscaling, health checks, monitoring, and recovery procedures.",
    ]
    rows = []
    for offset, prompt in enumerate(prompts):
        row_id = valid_control_id if offset == len(prompts) - 1 else f"cloud-{offset + 1:06d}"
        rows.append(
            {
                "id": row_id,
                "prompt": prompt,
                "reasoning": None,
                "response": repeated_response,
                "metadata": {
                    "category": "general_instruction_following",
                    "difficulty": 2,
                    "template_family": f"cloud_audit_case_{offset % 5}",
                    "eval_family": None,
                },
            }
        )
    cloud.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = build_response_diversity_summary([cloud])

    assert len(summary["repeated_response_clusters"]) == 1
    cluster = summary["repeated_response_clusters"][0]
    assert cluster["count"] == 29
    assert cluster["responses"] == [repeated_response]
    assert len(cluster["response_fingerprint"]) == 64
    assert len(cluster["members"]) == 29
    assert all(len(member["member_fingerprint"]) == 64 for member in cluster["members"])
    assert len({member["member_fingerprint"] for member in cluster["members"]}) == 29
    assert {member["id"] for member in cluster["members"]} == {row["id"] for row in rows}
    assert {member["prompt"] for member in cluster["members"]} == set(prompts)
    control = next(member for member in cluster["members"] if member["id"] == valid_control_id)
    assert control["prompt"] == prompts[-1]
    assert {member["signal"] for member in cluster["members"]} == {"cloud"}
    assert {member["template_family"] for member in cluster["members"]} == {
        f"cloud_audit_case_{index}" for index in range(5)
    }
