from slm_synth.adaptive_batch import AdaptiveBatchSizeController


def test_adaptive_batch_size_halves_on_failure_and_recovers_slowly():
    controller = AdaptiveBatchSizeController(maximum=32, minimum=1, increase_successes=2)

    controller.record_failure()
    controller.record_failure()

    assert controller.current == 8
    assert controller.snapshot()["adaptive_batch_size_decreases"] == 2
    assert controller.snapshot()["adaptive_batch_size_observed_minimum"] == 8

    controller.record_success()
    assert controller.current == 8

    controller.record_success()
    assert controller.current == 16
    assert controller.snapshot()["adaptive_batch_size_increases"] == 1
