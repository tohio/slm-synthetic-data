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

`target_total_tokens` is an estimated corpus-size planning target. It is divided by each signal's `avg_tokens_per_sample` estimate to derive target row counts, then rounded to complete 32-record request batches.

After a clean smoke run, update the `avg_tokens_per_sample` values using:

```bash
python -m slm_synth.report_lengths --config configs/synthetic.yaml --stage deduped
```

No downstream tokenizer is required by this repository.
