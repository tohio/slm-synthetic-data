from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from slm_synth.artifacts.arithmetic import ArithmeticArtifactFactory
from slm_synth.artifacts.base import GroundedArtifact
from slm_synth.record_quality import validate_record


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class CorpusTokenCounter:
    """Count tokens in the text representation intended for corpus ingestion."""

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, tokenizer_name_or_path: str) -> "CorpusTokenCounter":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        return cls(tokenizer)

    @staticmethod
    def arithmetic_text(record: dict[str, Any]) -> str:
        steps = record.get("steps") or []
        parts = [str(record.get("question", "")), *[str(step) for step in steps], str(record.get("answer", ""))]
        return "\n".join(part for part in parts if part)

    def count_arithmetic_records(self, records: list[dict[str, Any]]) -> int:
        return sum(
            len(self.tokenizer.encode(self.arithmetic_text(record), add_special_tokens=False))
            for record in records
        )


class GroundedBatchStore:
    """Persist completed rendered batches atomically and rebuild raw JSONL safely.

    A completed request is stored as one JSON file before the signal raw JSONL is
    materialized. A restart rebuilds raw JSONL from completed batch files, so an
    interrupted process cannot duplicate accepted rows by appending twice.
    """

    def __init__(self, output_dir: Path, signal: str):
        self.signal = signal
        self.batch_dir = output_dir / "manifests" / "grounded" / signal / "batches"
        self.raw_path = output_dir / "raw" / f"{signal}.jsonl"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_id: int) -> Path:
        return self.batch_dir / f"batch_{batch_id:09d}.json"

    def completed_batch_ids(self) -> list[int]:
        ids: list[int] = []
        for path in self.batch_dir.glob("batch_*.json"):
            ids.append(int(path.stem.split("_")[1]))
        return sorted(ids)

    def write_completed(
        self,
        *,
        batch_id: int,
        artifacts: list[GroundedArtifact],
        records: list[dict[str, Any]],
        token_count: int,
    ) -> None:
        payload = {
            "batch_id": batch_id,
            "signal": self.signal,
            "artifact_ids": [artifact.artifact_id for artifact in artifacts],
            "artifacts": [asdict(artifact) for artifact in artifacts],
            "records": records,
            "raw_token_count": int(token_count),
        }
        final_path = self._path(batch_id)
        temp_path = final_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, final_path)

    def total_tokens(self) -> int:
        return sum(self._load(path).get("raw_token_count", 0) for path in self._completed_paths())

    def materialize_raw(self) -> int:
        temp_path = self.raw_path.with_suffix(".tmp")
        rows = 0
        with temp_path.open("w", encoding="utf-8") as handle:
            for path in self._completed_paths():
                for record in self._load(path).get("records", []):
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.raw_path)
        return rows

    def _completed_paths(self) -> list[Path]:
        return [self._path(batch_id) for batch_id in self.completed_batch_ids()]

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


class GroundedArithmeticGenerator:
    """Render deterministic arithmetic artifacts through one structured LLM call."""

    def __init__(self, llm: Any, *, batch_size: int = 32, factory: ArithmeticArtifactFactory | None = None):
        self.llm = llm
        self.batch_size = int(batch_size)
        self.factory = factory or ArithmeticArtifactFactory()

    @staticmethod
    def response_schema(batch_size: int) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "minItems": batch_size,
                    "maxItems": batch_size,
                    "items": {
                        "type": "object",
                        "properties": {
                            "artifact_id": {"type": "string"},
                            "question": {"type": "string"},
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 5,
                            },
                        },
                        "required": ["artifact_id", "question", "steps"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["records"],
            "additionalProperties": False,
        }

    @staticmethod
    def build_prompt(artifacts: list[GroundedArtifact]) -> str:
        rows = [
            {
                "artifact_id": artifact.artifact_id,
                "family": artifact.family,
                "payload": artifact.payload,
            }
            for artifact in artifacts
        ]
        return (
            "Generate one arithmetic pretraining record for each grounded artifact below. "
            "The artifact is authoritative. Preserve each artifact_id exactly and return records in the same order. "
            "Generate only a natural learner-facing question and compact worked steps; the verified answer is held locally. "
            "Use every required_numeric_literal in the question, introduce no extra numeric quantities, and do not reveal "
            "the answer in the question. Return only the JSON object required by the schema.\n\n"
            "GROUNDED ARTIFACTS:\n" + json.dumps(rows, ensure_ascii=False, indent=2)
        )

    def generate_batch(self, batch_id: int) -> tuple[list[GroundedArtifact], list[dict[str, Any]]]:
        artifacts = self.factory.build_batch(batch_id, self.batch_size)
        rendered = self.llm.generate_structured_object(
            prompt=self.build_prompt(artifacts),
            schema=self.response_schema(self.batch_size),
            schema_name=f"grounded_arithmetic_batch_{self.batch_size}",
        )
        outputs = rendered.get("records") if isinstance(rendered, dict) else None
        if not isinstance(outputs, list) or len(outputs) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} grounded arithmetic records")

        expected = {artifact.artifact_id: artifact for artifact in artifacts}
        returned_ids = [row.get("artifact_id") for row in outputs if isinstance(row, dict)]
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(expected):
            raise ValueError("Grounded arithmetic response has missing, duplicate, or unexpected artifact IDs")

        records: list[dict[str, Any]] = []
        for row in outputs:
            artifact = expected[row["artifact_id"]]
            question = str(row.get("question", "")).strip()
            observed_numbers = re.findall(r"(?<![\w])-?\d+(?![\w])", question)
            required_numbers = list(artifact.payload.get("required_numeric_literals", []))
            if Counter(observed_numbers) != Counter(required_numbers):
                raise ValueError(
                    f"Rendered question changed numeric facts for {artifact.artifact_id}: "
                    f"expected={required_numbers}, observed={observed_numbers}"
                )
            if artifact.payload["answer"] in observed_numbers and artifact.payload["answer"] not in required_numbers:
                raise ValueError(f"Rendered question leaks answer for {artifact.artifact_id}")
            record = {
                "type": "arithmetic",
                "question": question,
                "steps": row.get("steps"),
                "answer": artifact.payload["answer"],
                "verification_expression": artifact.payload["expression"],
                "verification_answer": artifact.payload["answer"],
            }
            result = validate_record("arithmetic", record, require_arithmetic_verification=True)
            if not result.ok:
                raise ValueError(f"Rendered arithmetic record failed validation: {result.issues}")
            records.append(record)
        return artifacts, records
