import sys

import pytest
import yaml

from configs import configure_synthetic


def run_configure(monkeypatch, tmp_path, *arguments):
    output_path = tmp_path / "synthetic.yaml"
    monkeypatch.setattr(configure_synthetic, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["configure_synthetic.py", "--tokens", "250000", "--run", "qualification", *arguments],
    )
    configure_synthetic.main()
    return yaml.safe_load(output_path.read_text())


def test_configure_allows_max_qualification_batch_concurrency_and_output_tokens(monkeypatch, tmp_path):
    cfg = run_configure(
        monkeypatch,
        tmp_path,
        "--batch-size", "64",
        "--concurrency", "1024",
        "--max-tokens", "16384",
    )
    assert cfg["generation"]["batch_size"] == 64
    assert cfg["generation"]["parallel_requests"] == 1024
    assert cfg["backend"]["max_tokens"] == 16384
    assert all(signal["batch_size"] == 64 for signal in cfg["mix"].values())


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--batch-size", "65"), "--batch-size must be between 1 and 64"),
        (("--concurrency", "1025"), "--concurrency must be between 1 and 1024"),
    ],
)
def test_configure_rejects_throughput_values_above_qualification_limits(monkeypatch, tmp_path, arguments, message):
    with pytest.raises(ValueError, match=message):
        run_configure(monkeypatch, tmp_path, *arguments)
