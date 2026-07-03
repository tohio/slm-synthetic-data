# SLM Synthetic Data

Synthetic dataset generation for the SLM training stack.

| Dataset | Smoke Run | Target Run | Public Output |
|---|---|---|---|
| Pretraining synthetic data | `make pretrain-smoke` | `make pretrain-generate` | `data/runs/<run>/deduped` |
| Response distillation | `make distill-smoke` | `make distill-generate` | `data/distillation/runs/<run>/datasets` |
| SFT | `make sft-smoke` | `make sft-generate` | `data/sft/runs/<run>/datasets` |
| DPO | `make dpo-smoke` | `make dpo-generate` | `data/dpo/runs/<run>/datasets` |

OpenRouter is the only supported live provider. Provider calls, retries, concurrency, and structured-output handling live in `slm_synth/llm.py`.

## Setup

```bash
git clone https://github.com/tohio/slm-synthetic-data.git
cd slm-synthetic-data
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
OPENROUTER_API_KEY=...
HF_TOKEN=...       # only needed for Hugging Face publishing
```

The default live model is:

```bash
MODEL=openai/gpt-4.1-mini
```

Override it on any generation command:

```bash
make sft-smoke MODEL=<provider/model>
```

## Pretraining

Pretraining data is grounded synthetic text for continued pretraining or mixing into a broader pretraining corpus.

Supported signals:

```text
arithmetic
task_code
educational_qa_mcq_math
educational_qa_mcq_general
factual_restraint
```

Small run:

```bash
make pretrain-smoke
make pretrain-inspect
```

Target run:

```bash
make pretrain-generate \
  PRETRAIN_TARGET_TOKENS=1000000 \
  PRETRAIN_TARGET_CONCURRENCY=4

make pretrain-inspect PRETRAIN_INSPECT_RUN=pretrain-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `PRETRAIN_TOKENS` | `100000` | Smoke token target |
| `PRETRAIN_TARGET_TOKENS` | `1000000` | Target token target |
| `PRETRAIN_CONCURRENCY` | `1` | Smoke request concurrency |
| `PRETRAIN_TARGET_CONCURRENCY` | `4` | Target request concurrency |
| `PRETRAIN_INSPECT_RUN` | `$(PRETRAIN_REPORT_RUN)` | Run inspected by `pretrain-inspect` |
| `PRETRAIN_MODEL` | `$(MODEL)` | Pretraining model |
| `PRETRAIN_SIGNAL` | unset | Optional single-signal filter |

## Response Distillation

Distillation data is teacher prompt/response supervision.

Public row schema:

```json
{"id": "string", "prompt": "string", "reasoning": null, "response": "string"}
```

Supported signals:

```text
arithmetic
cloud
code
data_transform
database
debugging
educational_qa
factual_restraint
instruction
planning
```

Small run:

```bash
make distill-smoke
make distill-inspect
```

Target run:

```bash
make distill-generate \
  DISTILL_TARGET_SIZE=pilot \
  DISTILL_TARGET_RUN=distill-pilot-001

make distill-inspect DISTILL_INSPECT_RUN=distill-pilot-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `DISTILL_SMOKE_COUNT_PER_SIGNAL` | `2` | Smoke rows per signal |
| `DISTILL_TARGET_SIZE` | `pilot` | Target preset |
| `DISTILL_BATCH_SIZE` | `5` | Prompts per teacher request |
| `DISTILL_CONCURRENCY` | `1` | Parallel teacher requests |
| `DISTILL_RUN_ROOT` | `data/distillation/runs` | Run output root |
| `DISTILL_INSPECT_RUN` | `$(DISTILL_REPORT_RUN)` | Run inspected by `distill-inspect` |
| `DISTILL_SIGNALS` | unset | Optional signal list |
| `DISTILL_MODEL` | `$(MODEL)` | Teacher model |
| `DISTILL_MAX_TOKENS` | `4096` | Teacher response max tokens |

## SFT

SFT data is chat-style supervised instruction data.

Public row schema:

```json
{
  "id": "string",
  "messages": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "metadata": {
    "category": "string",
    "difficulty": 1,
    "template_family": "string",
    "eval_family": "string | null"
  }
}
```

Small run:

```bash
make sft-smoke
make sft-inspect
```

Target run:

```bash
make sft-generate \
  SFT_FAMILIES=all \
  SFT_TARGET_RUN=sft-target-001 \
  SFT_COUNT_PER_FAMILY=1000 \
  SFT_CONCURRENCY=2

make sft-inspect SFT_INSPECT_RUN=sft-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `SFT_SMOKE_FAMILIES` | `basic_arithmetic_qa` | Smoke family list |
| `SFT_FAMILIES` | `all` | Target family list |
| `SFT_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family |
| `SFT_COUNT_PER_FAMILY` | `1000` | Target rows per family |
| `SFT_BATCH_SIZE` | `5` | Specs per teacher request |
| `SFT_CONCURRENCY` | `1` | Request concurrency across run batches |
| `SFT_RUN_ROOT` | `data/sft/runs` | Run output root |
| `SFT_REPORT_RUN` | `$(SFT_RUN)` | Run used by `sft-report` |
| `SFT_INSPECT_RUN` | `$(SFT_REPORT_RUN)` | Run inspected by `sft-inspect` |
| `SFT_MODEL` | `$(MODEL)` | Teacher model |
| `SFT_MAX_TOKENS` | `4096` | Teacher response max tokens |

## DPO

DPO data is preference supervision with chosen and rejected assistant answers.

Public row schema:

```json
{
  "id": "string",
  "prompt": [{"role": "user", "content": "string"}],
  "chosen": [{"role": "assistant", "content": "string"}],
  "rejected": [{"role": "assistant", "content": "string"}],
  "metadata": {
    "category": "string",
    "difficulty": 1,
    "template_family": "string",
    "eval_family": "string | null",
    "failure_mode": "string"
  }
}
```

Small run:

```bash
make dpo-smoke
make dpo-inspect
```

Target run:

```bash
make dpo-generate \
  DPO_FAMILIES=all \
  DPO_TARGET_RUN=dpo-target-001 \
  DPO_COUNT_PER_FAMILY=1000 \
  DPO_CONCURRENCY=2

make dpo-inspect DPO_INSPECT_RUN=dpo-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `DPO_SMOKE_FAMILIES` | `basic_arithmetic_qa` | Smoke family list |
| `DPO_FAMILIES` | `all` | Target family list |
| `DPO_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family |
| `DPO_COUNT_PER_FAMILY` | `1000` | Target rows per family |
| `DPO_BATCH_SIZE` | `5` | Specs per teacher request |
| `DPO_CONCURRENCY` | `1` | Request concurrency across run batches |
| `DPO_RUN_ROOT` | `data/dpo/runs` | Run output root |
| `DPO_REPORT_RUN` | `$(DPO_RUN)` | Run used by `dpo-report` |
| `DPO_INSPECT_RUN` | `$(DPO_REPORT_RUN)` | Run inspected by `dpo-inspect` |
| `DPO_MODEL` | `$(MODEL)` | Teacher model |
| `DPO_MAX_TOKENS` | `4096` | Teacher response max tokens |

## Reports

Smoke and target generation commands build their reports automatically. Rebuild reports manually when needed:

```bash
make pretrain-report PRETRAIN_REPORT_RUN=<run-id>
make distill-report DISTILL_REPORT_RUN=<run-id>
make sft-report SFT_REPORT_RUN=<run-id>
make dpo-report DPO_REPORT_RUN=<run-id>
```

## Test

```bash
make test
```

## Command Reference

See [docs/COMMANDS.md](docs/COMMANDS.md) for the full Make variable reference.
