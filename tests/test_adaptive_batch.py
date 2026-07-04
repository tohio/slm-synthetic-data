from slm_synth.adaptive_batch import AdaptiveBatchSizeController


def test_adaptive_batch_size_starts_small_halves_on_failure_and_recovers_slowly():
    controller = AdaptiveBatchSizeController(maximum=32, minimum=1, increase_successes=2)

    assert controller.current == 4
    assert controller.snapshot()["adaptive_batch_size_observed_peak"] == 4

    controller.record_failure()
    controller.record_failure()

    assert controller.current == 1
    assert controller.snapshot()["adaptive_batch_size_decreases"] == 2
    assert controller.snapshot()["adaptive_batch_size_observed_minimum"] == 1

    controller.record_success()
    assert controller.current == 1

    controller.record_success()
    assert controller.current == 2
    assert controller.snapshot()["adaptive_batch_size_increases"] == 1


def test_adaptive_batch_size_doubles_to_maximum_after_stable_successes():
    controller = AdaptiveBatchSizeController(maximum=16, minimum=1, increase_successes=2)

    assert controller.current == 4

    controller.record_success()
    controller.record_success()
    assert controller.current == 8

    controller.record_success()
    controller.record_success()
    assert controller.current == 16

    controller.record_success()
    controller.record_success()
    assert controller.current == 16
    assert controller.snapshot()["adaptive_batch_size_observed_peak"] == 16


def test_adaptive_batch_size_default_growth_requires_stable_success_window():
    controller = AdaptiveBatchSizeController(maximum=64)

    for _ in range(15):
        controller.record_success()
    assert controller.current == 4

    controller.record_success()
    assert controller.current == 8


def test_adaptive_batch_size_can_start_aggressively_for_production_runs():
    controller = AdaptiveBatchSizeController(maximum=256, minimum=1, initial=64, increase_successes=4)

    assert controller.current == 64
    for _ in range(4):
        controller.record_success()
    assert controller.current == 128
    for _ in range(4):
        controller.record_success()
    assert controller.current == 256


def test_adaptive_batch_size_initial_is_capped_by_maximum_and_minimum():
    assert AdaptiveBatchSizeController(maximum=2).current == 2
    assert AdaptiveBatchSizeController(maximum=8, minimum=6).current == 6
