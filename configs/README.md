# Synthetic Configuration

`configs/configure_synthetic.py` writes `configs/synthetic.yaml` for the grounded pretraining generation path.

## Example

```bash
python configs/configure_synthetic.py \
  --profile balanced \
  --tokens 100000 \
  --batch-size 32 \
  --concurrency 4 \
  --model openai/gpt-5.6-luna-pro \
  --run pretrain-smoke-001
```

## Profiles

Profiles control request posture (batch size/concurrency) rather than dataset semantics. The generator model is configurable; the current pretraining production default is Luna Pro, with Gemma judge and Luna reviewer configured by the Makefile.

## Target sizing

`target_total_tokens` is the final accepted public-text target. Grounded generation produces candidates, deterministic validation filters them, semantic judge/reviewer stages accept high-quality rows, final global dedup removes duplicates, and post-review accepted-token accounting requests backfill through the same full path when required.

IDs and metadata do not count toward the target. Token accounting uses the configured character/token estimate because downstream tokenizer selection belongs to the training repository. `accepted_token_report.json` is the completion contract used by verification and publication.

Use `make pretrain-report` to inspect the final accepted dataset; the standalone legacy length-report command is not part of the supported workflow.
