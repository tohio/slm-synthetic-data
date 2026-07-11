from slm_synth.telemetry import aggregate_llm_telemetry


def test_aggregate_llm_telemetry_sums_nested_batch_counts():
    telemetry = aggregate_llm_telemetry(
        [
            {
                "batch_count": 2,
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "cost": 0.01},
                "retry_count": 1,
                "retryable_provider_retries": 2,
                "elapsed_seconds": 4.0,
                "routing_mode": "prefer",
                "requested_provider": "deepinfra",
                "allow_fallbacks": True,
            },
            {
                "batch_count": 3,
                "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18, "cost": 0.02},
                "retry_count": 4,
                "retryable_provider_retries": 5,
                "elapsed_seconds": 6.0,
            },
        ]
    )

    assert telemetry["batch_count"] == 5
    assert telemetry["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 16,
        "total_tokens": 26,
        "cost": 0.03,
    }
    assert telemetry["retry_count"] == 5
    assert telemetry["retryable_provider_retries"] == 7
    assert telemetry["aggregate_request_seconds"] == 10.0
    assert "elapsed_seconds" not in telemetry
    assert telemetry["routing_mode"] == "prefer"
    assert telemetry["requested_provider"] == "deepinfra"
    assert telemetry["allow_fallbacks"] is True


def test_aggregate_llm_telemetry_counts_raw_items_as_single_batches():
    telemetry = aggregate_llm_telemetry([{"usage": {"total_tokens": 2}}, {"usage": {"total_tokens": 3}}])

    assert telemetry["batch_count"] == 2
    assert telemetry["usage"]["total_tokens"] == 5


def test_aggregate_llm_telemetry_accepts_already_aggregated_request_seconds():
    telemetry = aggregate_llm_telemetry(
        [
            {"batch_count": 2, "aggregate_request_seconds": 10.5},
            {"batch_count": 3, "elapsed_seconds": 4.5},
        ]
    )

    assert telemetry["batch_count"] == 5
    assert telemetry["aggregate_request_seconds"] == 15.0
    assert "elapsed_seconds" not in telemetry
