from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def _as_paths(paths: Iterable[str | Path]) -> list[Path]:
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("at least one JSONL path is required")
    return resolved


def _read_jsonl_count(paths: Iterable[str | Path]) -> int:
    count = 0
    for path in _as_paths(paths):
        with path.open("r", encoding="utf-8") as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def _first_metadata(paths: Iterable[str | Path]) -> dict[str, object]:
    for path in _as_paths(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("metadata")
                if isinstance(metadata, dict):
                    return metadata
                return {}
    return {}


def _schema_for_kind(kind: str) -> str:
    if kind == "sft":
        lines = [
            "{",
            '  "id": "string",',
            '  "messages": [',
            '    {"role": "user", "content": "string"},',
            '    {"role": "assistant", "content": "string"}',
            "  ],",
            '  "metadata": {',
            '    "category": "string",',
            '    "difficulty": 1,',
            '    "template_family": "string",',
            '    "eval_family": "string"',
            "  }",
            "}",
        ]
        return "\n".join(lines)

    if kind == "dpo":
        lines = [
            "{",
            '  "id": "string",',
            '  "prompt": [',
            '    {"role": "user", "content": "string"}',
            "  ],",
            '  "chosen": [',
            '    {"role": "assistant", "content": "string"}',
            "  ],",
            '  "rejected": [',
            '    {"role": "assistant", "content": "string"}',
            "  ],",
            '  "metadata": {',
            '    "category": "string",',
            '    "difficulty": 1,',
            '    "template_family": "string",',
            '    "eval_family": "string",',
            '    "failure_mode": "string"',
            "  }",
            "}",
        ]
        return "\n".join(lines)

    raise ValueError(f"unsupported family dataset card kind: {kind}")


def build_family_dataset_card(*, kind: str, family: str, jsonl_paths: Iterable[str | Path]) -> str:
    """Build a README.md dataset card for one public SFT/DPO family repo."""
    if kind not in {"sft", "dpo"}:
        raise ValueError(f"unsupported family dataset card kind: {kind}")

    paths = _as_paths(jsonl_paths)
    row_count = _read_jsonl_count(paths)
    metadata = _first_metadata(paths)
    category = metadata.get("category")
    difficulty = metadata.get("difficulty")
    template_family = metadata.get("template_family")
    eval_family = metadata.get("eval_family") or family

    if kind == "sft":
        title = f"SLM Synthetic SFT — {family}"
        summary = (
            "Synthetic supervised fine-tuning dataset containing "
            f"prompt-response conversations for `{family}`."
        )
        dataset_type = "supervised fine-tuning"
        total_label = "Total rows"
        intended_use = "Use this dataset for supervised fine-tuning experiments targeting this signal."
        limitations = (
            "This dataset is synthetic. Inspect samples before training and do not use it "
            "as an evaluation benchmark."
        )
    else:
        title = f"SLM Synthetic DPO — {family}"
        summary = (
            "Synthetic preference-pair dataset containing chosen and rejected "
            f"assistant responses for `{family}`."
        )
        dataset_type = "preference optimization"
        total_label = "Total pairs"
        intended_use = "Use this dataset for DPO-style preference optimization experiments targeting this signal."
        limitations = (
            "This dataset is synthetic and encodes targeted preference behavior. "
            "Inspect chosen/rejected pairs before training."
        )

    metadata_lines = []
    if category is not None:
        metadata_lines.append(f"- Category: `{category}`")
    if difficulty is not None:
        metadata_lines.append(f"- Difficulty: `{difficulty}`")
    if template_family is not None:
        metadata_lines.append(f"- Template family: `{template_family}`")
    if eval_family is not None:
        metadata_lines.append(f"- Eval family: `{eval_family}`")

    metadata_block = ""
    if metadata_lines:
        metadata_block = "\n\n## Signal Metadata\n\n" + "\n".join(metadata_lines)

    schema = _schema_for_kind(kind)

    return (
        "---\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/*.jsonl\n"
        "---\n\n"
        f"# {title}\n\n"
        "## Summary\n\n"
        f"{summary}\n\n"
        "## Dataset\n\n"
        f"- Dataset type: `{dataset_type}`\n"
        f"- {total_label}: `{row_count}`\n"
        f"- Signal: `{family}`\n"
        f"- Language: English"
        f"{metadata_block}\n\n"
        "## Schema\n\n"
        "```json\n"
        f"{schema}\n"
        "```\n\n"
        "## Intended Use\n\n"
        f"{intended_use}\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
    )
