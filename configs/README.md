# Synthetic Configuration

`configs/configure_synthetic.py` writes `configs/synthetic.yaml` for the grounded OpenRouter generation path.

## Example

```bash
python configs/configure_synthetic.py \
  --profile balanced \
  --tokens 100000 \
  --batch-size 32 \
  --concurrency 4 \
  --run grounded_smoke
```

## Profiles

| Profile | Renderer | Default concurrency | Purpose |
|---|---|---:|---|
| `speed` | `deepseek/deepseek-v4-flash` | 8 | Throughput-focused runs after capacity is known |
| `balanced` | `deepseek/deepseek-v4-flash` | 4 | Recommended default |
| `quality` | `deepseek/deepseek-v4-flash` | 2 | Conservative request posture |

All profiles use the qualified DeepSeek renderer; profile differences affect runtime posture rather than corpus architecture.

## Target sizing

`target_total_tokens` is the accepted public-text target. The first candidate
round uses each signal's share and `avg_tokens_per_sample` estimate. After every
round, the pipeline validates and globally deduplicates the records, counts only
accepted public `text`, and requests unused replacement candidates for any
remaining per-signal token deficit.

IDs and metadata do not count toward the target. Token counts use the configured
`generation.chars_per_token` estimate because downstream tokenization belongs to
the training repository. If a genuine candidate inventory or an optional cost
ceiling is exhausted first, generation writes `accepted_token_report.json` and
exits nonzero rather than silently treating a short corpus as complete.

After a clean smoke run, update the `avg_tokens_per_sample` values using:

```bash
python -m slm_synth.pretrain.report_lengths --config configs/synthetic.yaml --stage deduped
```

This repository deliberately does not import a downstream model tokenizer.
