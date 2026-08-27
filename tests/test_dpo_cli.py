import pytest

from slm_synth.dpo.cli import main
from slm_synth.dpo.io import write_jsonl


def _sample_dpo_rows(count: int = 1) -> list[dict[str, object]]:
    return [
        {
            "id": f"dpo-sample-{index}",
            "prompt": [{"role": "user", "content": "What is 2 + 2? Answer with only the integer."}],
            "chosen": [{"role": "assistant", "content": "4"}],
            "rejected": [{"role": "assistant", "content": "The answer is 4."}],
            "metadata": {
                "task_family": "applied_math_and_reasoning",
                "interaction_modes": ["single_turn"],
                "output_mode": "concise",
                "context_mode": "self_contained",
                "preference_dimension": "factual_accuracy",
                "difficulty": 1,
                "template_family": "direct_qa",
                "failure_mode": "extra_explanation",
            },
        }
        for index in range(1, count + 1)
    ]


@pytest.mark.parametrize(
    "command", ["build-specs", "materialize-llm-batch", "generate-llm-batch", "generate-llm-run"]
)
def test_dpo_cli_has_no_standalone_legacy_generation_paths(command):
    with pytest.raises(SystemExit):
        main([command])


def test_dpo_report_coverage_cli_prints_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    write_jsonl(_sample_dpo_rows(), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path)]) == 0

    captured = capsys.readouterr()
    assert '"dataset_type": "dpo"' in captured.out
    assert '"row_count": 1' in captured.out
    assert '"preference_dimension_counts"' in captured.out
    assert '"factual_accuracy": 1' in captured.out
    assert '"extra_explanation": 1' in captured.out


def test_dpo_report_coverage_cli_writes_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    report_path = tmp_path / "coverage.json"
    write_jsonl(_sample_dpo_rows(), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path), "--output", str(report_path)]) == 0

    captured = capsys.readouterr()
    assert "wrote DPO coverage report" in captured.out
    assert report_path.exists()
