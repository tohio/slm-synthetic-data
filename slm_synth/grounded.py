from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from slm_synth.artifacts import (
    ArithmeticArtifactFactory,
    EducationalQAMCQGeneralArtifactFactory,
    EducationalQAMCQMathArtifactFactory,
    FactualRestraintArtifactFactory,
    GroundedArtifact,
    TaskCodeArtifactFactory,
)
from slm_synth.artifacts.quality import assert_valid_artifacts
from slm_synth.record_quality import validate_record


FACTORY_MAP = {
    "arithmetic": ArithmeticArtifactFactory,
    "task_code": TaskCodeArtifactFactory,
    "educational_qa_mcq_math": EducationalQAMCQMathArtifactFactory,
    "educational_qa_mcq_general": EducationalQAMCQGeneralArtifactFactory,
    "factual_restraint": FactualRestraintArtifactFactory,
}


class GroundedBatchStore:
    """Persist completed model requests atomically and rebuild raw JSONL safely."""

    def __init__(self, output_dir: Path, signal: str):
        self.signal = signal
        self.batch_dir = output_dir / "manifests" / "grounded" / signal / "batches"
        self.raw_path = output_dir / "raw" / f"{signal}.jsonl"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_id: int) -> Path:
        return self.batch_dir / f"batch_{batch_id:09d}.json"

    def completed_batch_ids(self) -> list[int]:
        return sorted(int(path.stem.split("_")[1]) for path in self.batch_dir.glob("batch_*.json"))

    def write_completed(
        self,
        *,
        batch_id: int,
        artifacts: list[GroundedArtifact],
        records: list[dict[str, Any]],
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "batch_id": int(batch_id),
            "signal": self.signal,
            "artifact_ids": [artifact.artifact_id for artifact in artifacts],
            "artifacts": [asdict(artifact) for artifact in artifacts],
            "records": records,
            "telemetry": telemetry or {},
        }
        final_path = self._path(batch_id)
        temp_path = final_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, final_path)

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

    def telemetry_summary(self) -> dict[str, float | int]:
        batches = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost = 0.0
        elapsed_seconds = 0.0
        retries = 0
        for path in self._completed_paths():
            telemetry = self._load(path).get("telemetry", {}) or {}
            usage = telemetry.get("usage", {}) or {}
            batches += 1
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            cost += float(usage.get("cost", 0.0) or 0.0)
            elapsed_seconds += float(telemetry.get("elapsed_seconds", 0.0) or 0.0)
            retries += int(telemetry.get("retry_count", 0) or 0)
        return {
            "batches": batches, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "total_tokens": total_tokens,
            "cost": cost, "elapsed_seconds": elapsed_seconds, "retry_count": retries,
        }


class GroundedSignalGenerator:
    """Render one homogeneous batch of deterministic grounded artifacts."""

    def __init__(self, signal: str, llm: Any, *, batch_size: int = 32, factory: Any | None = None):
        if signal not in FACTORY_MAP:
            raise ValueError(f"Unsupported grounded signal: {signal}")
        self.signal = signal
        self.llm = llm
        self.batch_size = int(batch_size)
        self.factory = factory or FACTORY_MAP[signal]()

    def response_schema(self) -> dict[str, Any]:
        common = {"artifact_id": {"type": "string"}}
        if self.signal == "arithmetic":
            fields = {**common, "question": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}}
            required = ["artifact_id", "question", "steps"]
        elif self.signal == "task_code":
            fields = {**common, "task": {"type": "string"}, "plan": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4}}
            required = ["artifact_id", "task", "plan"]
        elif self.signal == "educational_qa_mcq_math":
            fields = {**common, "question": {"type": "string"}, "explanation": {"type": "string"}}
            required = ["artifact_id", "question", "explanation"]
        elif self.signal == "educational_qa_mcq_general":
            fields = {**common, "explanation": {"type": "string"}}
            required = ["artifact_id", "explanation"]
        else:
            fields = {**common, "safe_answer": {"type": "string"}}
            required = ["artifact_id", "safe_answer"]
        item = {"type": "object", "properties": fields, "required": required, "additionalProperties": False}
        return {
            "type": "object",
            "properties": {"records": {"type": "array", "items": item, "minItems": self.batch_size, "maxItems": self.batch_size}},
            "required": ["records"],
            "additionalProperties": False,
        }

    def build_prompt(self, artifacts: list[GroundedArtifact]) -> str:
        rows = [{"artifact_id": item.artifact_id, "family": item.family, "payload": item.payload} for item in artifacts]
        common = (
            "Generate one final synthetic PRETRAINING record component for each grounded artifact below. "
            "Every grounded artifact is authoritative. Preserve artifact_id exactly and return records in the same order. "
            "Return only the JSON object required by the schema.\n\n"
        )
        instructions = {
            "arithmetic": (
                "For each artifact, generate a natural learner-facing question and compact worked steps. "
                "Use every required_numeric_literal in the question, introduce no extra numeric quantities, "
                "and do not reveal the held answer in the question. The verified answer remains local."
            ),
            "task_code": (
                "For each valid Python code artifact, generate a faithful task and a short 2-to-4 step plan. "
                "The task must start with 'Write a Python function that', state input/output behavior and non-mutation, "
                "and must not contain code, the held function name, or behavior absent from the supplied code."
            ),
            "educational_qa_mcq_math": (
                "For each artifact, generate a self-contained natural math question and concise explanation. "
                "Preserve all required numeric literals in the question. If required_text_literals are supplied, "
                "preserve each supplied term in the question exactly; do not collapse a word-problem item into a "
                "generic bare-number question. Choices and verified answer remain local."
            ),
            "educational_qa_mcq_general": (
                "For each artifact, generate only a concise explanation showing why the held answer follows from "
                "the supplied evidence. Evidence, question, choices, and answer remain local and must not be changed."
            ),
            "factual_restraint": (
                "For each artifact, generate a concise natural user-facing assistant answer to the supplied question. "
                "Follow the behavior requirement without inventing facts or writing policy instructions such as "
                "'Do not provide'. Use direct language such as 'I can\'t provide' or 'I can\'t verify'."
            ),
        }[self.signal]
        return common + instructions + "\n\nGROUNDED ARTIFACTS:\n" + json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _numeric_literals(text: str) -> list[str]:
        return re.findall(r"(?<![\w])-?\d+(?![\w])", text)

    def _finalize(self, artifact: GroundedArtifact, row: dict[str, Any]) -> dict[str, Any]:
        payload = artifact.payload
        if self.signal == "arithmetic":
            question = str(row.get("question", "")).strip()
            observed = self._numeric_literals(question)
            required = list(payload["required_numeric_literals"])
            if Counter(observed) != Counter(required):
                raise ValueError(f"Rendered arithmetic question changed numeric facts for {artifact.artifact_id}")
            if payload["answer"] in observed and payload["answer"] not in required:
                raise ValueError(f"Rendered arithmetic question leaks answer for {artifact.artifact_id}")
            record = {
                "type": "arithmetic", "question": question, "steps": row.get("steps"), "answer": payload["answer"],
                "verification_expression": payload["expression"], "verification_answer": payload["answer"],
            }
            result = validate_record("arithmetic", record, require_arithmetic_verification=True)
        elif self.signal == "task_code":
            task = str(row.get("task", "")).strip()
            lower = task.lower()
            if not lower.startswith("write a python function that") or "```" in task or "\ndef " in lower:
                raise ValueError(f"Rendered task_code task is not a clean instruction for {artifact.artifact_id}")
            record = {"type": "task_code", "task": task, "plan": row.get("plan"), "code": payload["code"]}
            result = validate_record("task_code", record)
        elif self.signal == "educational_qa_mcq_math":
            question = str(row.get("question", "")).strip()
            if Counter(self._numeric_literals(question)) != Counter(payload["required_numeric_literals"]):
                raise ValueError(f"Rendered math MCQ question changed numeric facts for {artifact.artifact_id}")
            record = {
                "type": "educational_qa_mcq_math", "question": question, "choices": payload["choices"],
                "correct_index": payload["correct_index"], "explanation": row.get("explanation"),
                "verification_expression": payload["expression"], "verification_answer": payload["answer"],
            }
            result = validate_record("educational_qa_mcq_math", record, require_mcq_verification=True)
        elif self.signal == "educational_qa_mcq_general":
            record = {
                "type": "educational_qa_mcq_general", "evidence": payload["evidence"],
                "question": payload["question"], "choices": payload["choices"],
                "correct_index": payload["correct_index"], "explanation": row.get("explanation"),
            }
            result = validate_record("educational_qa_mcq_general", record)
        else:
            record = {"type": "factual_restraint", "question": payload["question"], "safe_answer": row.get("safe_answer")}
            result = validate_record("factual_restraint", record)
        if not result.ok:
            raise ValueError(f"Rendered {self.signal} record failed validation for {artifact.artifact_id}: {result.issues}")
        return record

    def generate_batch(self, batch_id: int) -> tuple[list[GroundedArtifact], list[dict[str, Any]], dict[str, Any]]:
        artifacts = self.factory.build_batch(batch_id, self.batch_size)
        assert_valid_artifacts(artifacts)
        prompt = self.build_prompt(artifacts)
        if hasattr(self.llm, "generate_structured_object_with_metadata"):
            result = self.llm.generate_structured_object_with_metadata(
                prompt=prompt, schema=self.response_schema(),
                schema_name=f"grounded_{self.signal}_batch_{self.batch_size}",
            )
            response = result["data"]
            telemetry = result.get("telemetry", {})
        else:
            response = self.llm.generate_structured_object(
                prompt=prompt, schema=self.response_schema(),
                schema_name=f"grounded_{self.signal}_batch_{self.batch_size}",
            )
            telemetry = {}
        rows = response.get("records") if isinstance(response, dict) else None
        if not isinstance(rows, list) or len(rows) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} grounded {self.signal} records")
        expected = {artifact.artifact_id: artifact for artifact in artifacts}
        returned_ids = [row.get("artifact_id") for row in rows if isinstance(row, dict)]
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(expected):
            raise ValueError(f"Grounded {self.signal} response has missing, duplicate, or unexpected artifact IDs")
        return artifacts, [self._finalize(expected[row["artifact_id"]], row) for row in rows], telemetry
