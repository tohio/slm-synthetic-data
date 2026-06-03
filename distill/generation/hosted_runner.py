from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.generation.prompts import PromptRecord
from distill.providers.base import GenerationRequest, GenerationResponse


@dataclass(frozen=True)
class HostedGenerationResult:
    output_path: str
    written: int
    skipped: int
    errors: int


@dataclass(frozen=True)
class BatchGenerationItem:
    prompt: PromptRecord
    output: str
    response: GenerationResponse | None
    error: str | None = None


class HostedBatchStore:
    def __init__(self, output_path: Path, run_key: str):
        self.output_path = output_path
        self.batch_dir = Path('runs') / 'hosted_generation' / run_key / 'batches'
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_id: int) -> Path:
        return self.batch_dir / f'batch_{batch_id:09d}.json'

    def completed_batch_ids(self) -> set[int]:
        return {int(path.stem.split('_')[1]) for path in self.batch_dir.glob('batch_*.json')}

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_suffix('.tmp')
        with temp.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)

    def write_completed(self, *, batch_id: int, prompt_ids: list[str], records: list[dict[str, Any]], telemetry: dict[str, Any]) -> None:
        self._atomic_json(self._path(batch_id), {'batch_id': int(batch_id), 'prompt_ids': prompt_ids, 'records': records, 'telemetry': telemetry})

    def materialize_raw(self) -> int:
        temp = self.output_path.with_suffix('.tmp')
        rows = 0
        with temp.open('w', encoding='utf-8') as handle:
            for path in sorted(self.batch_dir.glob('batch_*.json')):
                for record in self._load(path).get('records', []):
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
                    rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.output_path)
        return rows

    def telemetry_summary(self) -> dict[str, Any]:
        result = {'batches': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'retry_count': 0, 'retryable_provider_retries': 0, 'adaptive_peak_in_flight_limit': 0, 'adaptive_window_increases': 0, 'adaptive_window_decreases': 0}
        for path in sorted(self.batch_dir.glob('batch_*.json')):
            result['batches'] += 1
            telemetry = self._load(path).get('telemetry', {}) or {}
            for key in ('prompt_tokens', 'completion_tokens', 'retry_count', 'retryable_provider_retries', 'adaptive_window_increases', 'adaptive_window_decreases'):
                result[key] += int(telemetry.get(key, 0) or 0)
            result['adaptive_peak_in_flight_limit'] = max(result['adaptive_peak_in_flight_limit'], int(telemetry.get('adaptive_peak_in_flight_limit', 0) or 0))
        return result


def build_batch_prompt(prompts: list[PromptRecord]) -> str:
    items = [{'id': p.id, 'category': p.category, 'prompt': p.prompt} for p in prompts]
    return ('Generate one supervised distillation output for each input item below. Preserve each id exactly and return items in the same order. Return only a JSON object with key `items`. The object must match this shape: {"items":[{"id":"...","output":"..."}]}. Each output must be concise, complete, directly useful, and free of unrelated filler.\n\nINPUT ITEMS:\n' + json.dumps(items, ensure_ascii=False, indent=2))


def _chunked(items: list[PromptRecord], size: int) -> list[list[PromptRecord]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _extract_json_value(text: str) -> Any:
    raw = text.strip()
    if raw.startswith('```'):
        raw = raw.removeprefix('```json').removeprefix('```').strip()
    if raw.endswith('```'):
        raw = raw.rsplit('```', 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        object_start, object_end = raw.find('{'), raw.rfind('}')
        array_start, array_end = raw.find('['), raw.rfind(']')
        if object_start != -1 and object_end > object_start:
            return json.loads(raw[object_start:object_end + 1])
        if array_start != -1 and array_end > array_start:
            return json.loads(raw[array_start:array_end + 1])
        raise


def parse_batch_response(*, prompts: list[PromptRecord], response_text: str) -> dict[str, str]:
    parsed = _extract_json_value(response_text)
    items = parsed.get('items') if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else None
    if not isinstance(items, list):
        raise ValueError('batch response must be a JSON object with items array')
    if len(items) != len(prompts):
        raise ValueError(f'batch response item count mismatch: expected={len(prompts)}, got={len(items)}')
    outputs: dict[str, str] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f'batch response item {idx} must be an object')
        expected = prompts[idx].id
        if item.get('id') != expected:
            raise ValueError(f'batch response id mismatch at index {idx}: expected={expected}, got={item.get("id")}')
        output = item.get('output')
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f'batch response item {idx} has empty output')
        outputs[expected] = output.strip()
    return outputs


def _record(*, prompt: PromptRecord, output: str, teacher_model: str, provider_name: str, response: GenerationResponse | None, error: str | None) -> dict[str, Any]:
    return {'id': prompt.id, 'prompt_id': prompt.id, 'category': prompt.category, 'prompt': prompt.prompt, 'teacher': teacher_model, 'teacher_model': teacher_model, 'teacher_provider': provider_name, 'provider': provider_name, 'model': teacher_model, 'response': output, 'output': output, 'input_tokens': response.input_tokens if response else None, 'output_tokens': response.output_tokens if response else None, 'metadata': prompt.metadata, 'raw': response.raw if response else None, 'error': error}


def _generate_batch_once(*, provider: Any, prompts: list[PromptRecord], teacher_model: str, provider_name: str, max_output_tokens: int, temperature: float, top_p: float) -> list[BatchGenerationItem]:
    if getattr(provider, 'provider_name', '') == 'dry_run':
        return [BatchGenerationItem(prompt=p, output=f'Dry run response for: {p.prompt}', response=GenerationResponse(text='{}', model=teacher_model, provider=provider_name, input_tokens=len(p.prompt.split()), output_tokens=8, raw={'dry_run': True})) for p in prompts]
    response = provider.generate(GenerationRequest(prompt=build_batch_prompt(prompts), model=teacher_model, max_output_tokens=max_output_tokens * len(prompts), temperature=temperature, top_p=top_p))
    try:
        outputs = parse_batch_response(prompts=prompts, response_text=response.text)
    except Exception:
        if len(prompts) != 1:
            raise
        plain = response.text.strip()
        if not plain:
            raise
        outputs = {prompts[0].id: plain}
    return [BatchGenerationItem(prompt=p, output=outputs[p.id], response=response) for p in prompts]


def _generate_batch_with_split(**kwargs: Any) -> list[BatchGenerationItem]:
    prompts = kwargs['prompts']
    min_batch_size = kwargs['min_batch_size']
    continue_on_error = kwargs['continue_on_error']
    try:
        call_kwargs = {k: v for k, v in kwargs.items() if k not in {'min_batch_size', 'continue_on_error'}}
        return _generate_batch_once(**call_kwargs)
    except Exception as exc:
        if len(prompts) > min_batch_size:
            mid = len(prompts) // 2
            left = dict(kwargs, prompts=prompts[:mid])
            right = dict(kwargs, prompts=prompts[mid:])
            return _generate_batch_with_split(**left) + _generate_batch_with_split(**right)
        if not continue_on_error:
            raise
        return [BatchGenerationItem(prompt=p, output='', response=None, error=str(exc)) for p in prompts]


def _telemetry(items: list[BatchGenerationItem]) -> dict[str, Any]:
    out = {'prompt_tokens': 0, 'completion_tokens': 0, 'retry_count': 0, 'retryable_provider_retries': 0, 'adaptive_peak_in_flight_limit': 0, 'adaptive_window_increases': 0, 'adaptive_window_decreases': 0}
    for item in items:
        if item.response is None:
            continue
        out['prompt_tokens'] += int(item.response.input_tokens or 0)
        out['completion_tokens'] += int(item.response.output_tokens or 0)
        raw = item.response.raw if isinstance(item.response.raw, dict) else {}
        telemetry = raw.get('telemetry', {}) if isinstance(raw, dict) else {}
        for key in ('retry_count', 'retryable_provider_retries', 'adaptive_window_increases', 'adaptive_window_decreases'):
            out[key] += int(telemetry.get(key, 0) or 0)
        out['adaptive_peak_in_flight_limit'] = max(out['adaptive_peak_in_flight_limit'], int(telemetry.get('adaptive_peak_in_flight_limit', 0) or 0))
    out['total_tokens'] = out['prompt_tokens'] + out['completion_tokens']
    return out


def _control(controls: Any, name: str, default: Any) -> Any:
    return getattr(controls, name, default) if controls is not None else default


def _run_key(provider_name: str, teacher_model: str, output_path: Path) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in teacher_model)
    return f'{provider_name}_{safe}_{output_path.stem}'


def run_hosted_generation(*, provider: Any, prompts: list[PromptRecord], output_path: str | Path, teacher_model: str, provider_name: str, max_output_tokens: int, temperature: float, top_p: float, controls: Any, continue_on_error: bool, batch_size: int | None = None, min_batch_size: int | None = None, parallel_requests: int | None = None, progress_interval: int | None = None, resume: bool = True) -> HostedGenerationResult:
    output = Path(output_path)
    effective_batch_size = int(batch_size or _control(controls, 'batch_size', 1) or 1)
    effective_min_batch_size = int(min_batch_size or _control(controls, 'min_batch_size', 1) or 1)
    effective_parallel = int(parallel_requests or _control(controls, 'parallel_requests', None) or _control(controls, 'concurrency', 1) or 1)
    effective_progress_interval = int(progress_interval or _control(controls, 'progress_interval', 25) or 25)
    all_batches = list(enumerate(_chunked(prompts, effective_batch_size)))
    store = HostedBatchStore(output, _run_key(provider_name, teacher_model, output))
    completed_ids = store.completed_batch_ids() if resume else set()
    if completed_ids and resume:
        store.materialize_raw()
    pending = [(batch_id, batch) for batch_id, batch in all_batches if batch_id not in completed_ids]
    skipped = sum(len(batch) for batch_id, batch in all_batches if batch_id in completed_ids)
    total = sum(len(batch) for _, batch in pending)
    if total == 0:
        return HostedGenerationResult(str(output), 0, skipped, 0)
    print(f'Hosted generation plan: prompts={len(prompts)} skipped={skipped} remaining={total} batches={len(pending)}/{len(all_batches)} batch_size={effective_batch_size} min_batch_size={effective_min_batch_size} parallel_requests={effective_parallel}', flush=True)
    written = errors = completed = 0
    next_progress = effective_progress_interval
    start = time.time()
    with ThreadPoolExecutor(max_workers=effective_parallel) as executor:
        futures = {executor.submit(_generate_batch_with_split, provider=provider, prompts=batch, teacher_model=teacher_model, provider_name=provider_name, max_output_tokens=max_output_tokens, temperature=temperature, top_p=top_p, min_batch_size=effective_min_batch_size, continue_on_error=continue_on_error): (batch_id, batch) for batch_id, batch in pending}
        for batch_index, future in enumerate(as_completed(futures), start=1):
            batch_id, batch = futures[future]
            items = future.result()
            records = []
            for item in items:
                written += 1
                completed += 1
                if item.error:
                    errors += 1
                records.append(_record(prompt=item.prompt, output=item.output, teacher_model=teacher_model, provider_name=provider_name, response=item.response, error=item.error))
            store.write_completed(batch_id=batch_id, prompt_ids=[p.id for p in batch], records=records, telemetry=_telemetry(items))
            if completed >= next_progress or completed >= total:
                elapsed = max(0.001, time.time() - start)
                metrics = store.telemetry_summary()
                print(f'Hosted generation progress: batches={batch_index}/{len(pending)} last_batch_records={len(items)} completed={completed}/{total} written={written} errors={errors} skipped={skipped} elapsed_seconds={elapsed:.1f} records_per_second={completed / elapsed:.2f} prompt_tokens={metrics["prompt_tokens"]} completion_tokens={metrics["completion_tokens"]} retryable_provider_retries={metrics["retryable_provider_retries"]} adaptive_peak_in_flight_limit={metrics["adaptive_peak_in_flight_limit"]}', flush=True)
                while next_progress <= completed:
                    next_progress += effective_progress_interval
    total_materialized = store.materialize_raw()
    elapsed = max(0.001, time.time() - start)
    metrics = store.telemetry_summary()
    print(f'Hosted generation completed: records={total_materialized} new_records={written} errors={errors} elapsed_seconds={elapsed:.1f} records_per_second={written / elapsed:.2f} prompt_tokens={metrics["prompt_tokens"]} completion_tokens={metrics["completion_tokens"]} retry_count={metrics["retry_count"]} retryable_provider_retries={metrics["retryable_provider_retries"]} adaptive_window_increases={metrics["adaptive_window_increases"]} adaptive_window_decreases={metrics["adaptive_window_decreases"]} adaptive_peak_in_flight_limit={metrics["adaptive_peak_in_flight_limit"]}', flush=True)
    return HostedGenerationResult(str(output), written, skipped, errors)
