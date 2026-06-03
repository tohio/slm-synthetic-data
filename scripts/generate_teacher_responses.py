from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from distill.generation.hosted_backend import HostedLLMBackend
from distill.generation.hosted_controls import hosted_controls_from_mapping
from distill.generation.hosted_runner import run_hosted_generation
from distill.generation.prompts import load_merged_prompt_records
from distill.generation.token_budget import select_prompts_for_token_target
from distill.providers.base import GenerationResponse
from distill.utils.config import load_response_distill_config, load_teachers_config


class DryRunProvider:
    provider_name = 'dry_run'
    def generate(self, request: Any) -> GenerationResponse:
        return GenerationResponse(text='{}', model=request.model, provider=self.provider_name, input_tokens=len(request.prompt.split()), output_tokens=8, raw={'dry_run': True})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate raw teacher responses.')
    parser.add_argument('--config', default='configs/response_distill_openrouter.yaml')
    parser.add_argument('--teachers', default='configs/teachers.yaml')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--target-tokens', type=int, default=None)
    parser.add_argument('--estimated-tokens-per-record', type=int, default=256)
    parser.add_argument('--allow-repeat-prompts', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--min-batch-size', type=int, default=None)
    parser.add_argument('--parallel-requests', type=int, default=None)
    parser.add_argument('--progress-interval', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def _teacher_map(teachers_config: Any) -> dict[str, Any]:
    candidate = getattr(teachers_config, 'teachers', None)
    if isinstance(candidate, dict):
        return candidate
    if isinstance(teachers_config, dict):
        raw = teachers_config.get('teachers') or teachers_config
        if isinstance(raw, dict):
            return raw
    raise TypeError('Unable to resolve teacher registry mapping')


def _teacher_field(teacher: Any, field: str) -> Any:
    return teacher[field] if isinstance(teacher, dict) else getattr(teacher, field)


def _provider_generation_config(config_path: str, provider_name: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding='utf-8')) or {}
    return ((raw.get('providers', {}) or {}).get(provider_name, {}) or {}).get('generation', {}) or {}


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.target_tokens is not None:
        raise SystemExit('Use either --limit or --target-tokens, not both.')
    run_config = load_response_distill_config(args.config)
    teacher = _teacher_map(load_teachers_config(args.teachers))[run_config.teacher_name]
    teacher_provider = _teacher_field(teacher, 'provider')
    teacher_model = _teacher_field(teacher, 'model')
    prompts = load_merged_prompt_records(run_config.data.prompts_paths)
    if args.limit is not None:
        prompts = prompts[:args.limit]
    prompts, token_plan = select_prompts_for_token_target(prompts, target_tokens=args.target_tokens, estimated_tokens_per_record=args.estimated_tokens_per_record, allow_repeat=args.allow_repeat_prompts)
    if token_plan is not None:
        print(f'Token target plan: target_tokens={token_plan.target_tokens} estimated_tokens_per_record={token_plan.estimated_tokens_per_record} selected_records={token_plan.selected_records} estimated_total_tokens={token_plan.estimated_total_tokens} repeated={token_plan.repeated}')
    generation_cfg = _provider_generation_config(args.config, teacher_provider)
    if args.batch_size is not None:
        generation_cfg['batch_size'] = args.batch_size
    if args.min_batch_size is not None:
        generation_cfg['min_batch_size'] = args.min_batch_size
    if args.parallel_requests is not None:
        generation_cfg['parallel_requests'] = args.parallel_requests
    if args.progress_interval is not None:
        generation_cfg['progress_interval'] = args.progress_interval
    provider = DryRunProvider() if args.dry_run else HostedLLMBackend.from_mapping(teacher_provider, generation_cfg)
    controls = hosted_controls_from_mapping(generation_cfg)
    result = run_hosted_generation(provider=provider, prompts=prompts, output_path=run_config.data.raw_teacher_path, teacher_model=teacher_model, provider_name=teacher_provider, max_output_tokens=run_config.distillation.max_output_tokens, temperature=run_config.distillation.temperature, top_p=run_config.distillation.top_p, controls=controls, continue_on_error=run_config.distillation.continue_on_error, batch_size=generation_cfg.get('batch_size'), min_batch_size=generation_cfg.get('min_batch_size'), parallel_requests=generation_cfg.get('parallel_requests'), progress_interval=generation_cfg.get('progress_interval'), resume=not args.no_resume)
    print(f'Wrote raw teacher records: {result.written}')
    if result.skipped:
        print(f'Skipped existing records: {result.skipped}')
    if result.errors:
        print(f'Generation errors: {result.errors}')
    print(f'Output: {result.output_path}')


if __name__ == '__main__':
    main()
