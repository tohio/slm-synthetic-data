# `slm_synth` package

## Production path

| Module | Purpose |
|---|---|
| `artifacts/` | Deterministic grounded artifacts and artifact preflight quality checks |
| `grounded.py` | Batch-of-32 structured rendering, anchored finalization and atomic persistence |
| `llm.py` | OpenRouter/DeepSeek client, structured JSON schema, retries and telemetry |
| `generate.py` | Orchestration, bounded concurrency and resume |
| `report_artifacts.py` | Artifact duplicate, family and quality reporting |
| `report_lengths.py` | Per-record size estimation for row-target calibration |
| `validate.py` | Final record validation |
| `dedup.py` | Exact deduplication |

Legacy two-pass sources remain present for backward compatibility with old configurations; generated grounded configurations route all five supported signals through `GroundedSignalGenerator`.
