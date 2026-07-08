# SLM Synthetic Data

Synthetic dataset generation for the SLM training stack.

| Dataset | Smoke Run | Target Run | Public Output |
|---|---|---|---|
| Pretraining synthetic data | `make pretrain-smoke` | `make pretrain-generate` | `data/runs/<run>/deduped` |
| Distillation SFT | `make distillation-sft-smoke` | `make distillation-sft-generate` | `data/distillation/runs/<run>/datasets` |
| Distillation DPO | `make distillation-dpo-smoke` | `make distillation-dpo-generate` | `data/distillation-dpo/runs/<run>/datasets` |
| SFT | `make sft-smoke` | `make sft-generate` | `data/sft/runs/<run>/datasets` |
| DPO | `make dpo-smoke` | `make dpo-generate` | `data/dpo/runs/<run>/datasets` |

OpenRouter is the only supported live provider. Provider calls, retries, concurrency, adaptive batch sizing, and structured-output handling live in `slm_synth/llm.py` and `slm_synth/adaptive_batch.py`.

Generation commands use the same throughput rules:

| Setting | Meaning |
|---|---|
| `*_CONCURRENCY` | Maximum simultaneous provider requests. Adaptive request admission may run below this cap when the provider throttles. |
| `*_BATCH_SIZE` | Maximum rows, prompts, or specs per provider request. Adaptive batch sizing halves failing batches and slowly increases after successful batches. |

OpenRouter routing defaults to `auto`. Use `prefer` to try one provider first while allowing fallback, or `strict` to require one provider. `prefer` and `strict` require `OPENROUTER_PROVIDER`.

| Variable | Default | Purpose |
|---|---:|---|
| `OPENROUTER_ROUTING_MODE` | `auto` | Routing policy: `auto`, `prefer`, or `strict` |
| `OPENROUTER_PROVIDER` | unset | Provider slug used by `prefer` or `strict` routing |

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

## Distillation SFT

Distillation SFT data is teacher prompt/response supervision. Smoke runs use tiny built-in seeds; target runs use deterministic production prompt specs.

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
make distillation-sft-smoke
make distillation-sft-inspect
```

Target run:

```bash
make distillation-sft-generate \
  DISTILLATION_SFT_TARGET_ROWS=100000 \
  DISTILLATION_SFT_TARGET_RUN=distillation-sft-100k-001

make distillation-sft-inspect DISTILLATION_SFT_INSPECT_RUN=distillation-sft-100k-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `DISTILLATION_SFT_SMOKE_COUNT_PER_SIGNAL` | `2` | Smoke rows per signal |
| `DISTILLATION_SFT_TARGET_ROWS` | `100000` | Target accepted public rows for production planning |
| `DISTILLATION_SFT_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum prompts per teacher request |
| `DISTILLATION_SFT_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests |
| `DISTILLATION_SFT_TARGET_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Target parallel teacher requests |
| `DISTILLATION_SFT_RUN_ROOT` | `data/distillation/runs` | Run output root |
| `DISTILLATION_SFT_INSPECT_RUN` | `$(DISTILLATION_SFT_REPORT_RUN)` | Run inspected by `distillation-sft-inspect` |
| `DISTILLATION_SFT_SIGNALS` | unset | Optional signal list |
| `DISTILLATION_SFT_MODEL` | `$(MODEL)` | Teacher model |
| `DISTILLATION_SFT_MAX_TOKENS` | `4096` | Teacher response max tokens |

## Distillation DPO

Distillation DPO data is preference data for aligning distilled models. It is isolated from generic DPO data. Target runs use teacher-quality chosen responses and controlled-weak rejected responses; student-model sampling is not part of this repo.

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
make distillation-dpo-smoke
make distillation-dpo-inspect
```

Target run:

```bash
make distillation-dpo-generate \
  DISTILLATION_DPO_TARGET_RUN=distillation-dpo-target-001 \
  DISTILLATION_DPO_TARGET_PAIRS=50000

make distillation-dpo-inspect DISTILLATION_DPO_INSPECT_RUN=distillation-dpo-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family |
| `DISTILLATION_DPO_TARGET_PAIRS` | `50000` | Target accepted preference pairs |
| `DISTILLATION_DPO_FAMILIES` | `all` | Target family list |
| `DISTILLATION_DPO_RUN_ROOT` | `data/distillation-dpo/runs` | Run output root |
| `DISTILLATION_DPO_INSPECT_RUN` | `$(DISTILLATION_DPO_REPORT_RUN)` | Run inspected by `distillation-dpo-inspect` |
| `DISTILLATION_DPO_MODEL` | `$(MODEL)` | Teacher lineage model recorded in manifests |
| `DISTILLATION_DPO_HF_NAMESPACE` | `$(HF_NAMESPACE)` | Hugging Face owner for push |
| `DISTILLATION_DPO_HF_PREFIX` | `distillation-dpo` | Hugging Face repo prefix |

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
  SFT_TARGET_ROWS=14000 \
  SFT_TARGET_CONCURRENCY=4

make sft-inspect SFT_INSPECT_RUN=sft-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `SFT_SMOKE_FAMILIES` | `basic_arithmetic_qa` | Smoke family list |
| `SFT_FAMILIES` | `all` | Target family list |
| `SFT_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family |
| `SFT_TARGET_ROWS` | `14000` | Target production rows across selected families |
| `SFT_COUNT_PER_FAMILY` | `1000` | Explicit low-level rows-per-family override |
| `SFT_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum specs per teacher request |
| `SFT_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests |
| `SFT_TARGET_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Target parallel teacher requests |
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
  DPO_TARGET_PAIRS=14000 \
  DPO_TARGET_CONCURRENCY=4

make dpo-inspect DPO_INSPECT_RUN=dpo-target-001
```

Useful variables:

| Variable | Default | Purpose |
|---|---:|---|
| `DPO_SMOKE_FAMILIES` | `basic_arithmetic_qa` | Smoke family list |
| `DPO_FAMILIES` | `all` | Target family list |
| `DPO_SMOKE_COUNT_PER_FAMILY` | `2` | Smoke rows per family |
| `DPO_TARGET_PAIRS` | `14000` | Target production pairs across selected families |
| `DPO_COUNT_PER_FAMILY` | `1000` | Explicit low-level pairs-per-family override |
| `DPO_BATCH_SIZE` | `$(PRETRAIN_BATCH_SIZE)` | Maximum specs per teacher request |
| `DPO_CONCURRENCY` | `$(PRETRAIN_CONCURRENCY)` | Smoke parallel teacher requests |
| `DPO_TARGET_CONCURRENCY` | `$(PRETRAIN_TARGET_CONCURRENCY)` | Target parallel teacher requests |
| `DPO_RUN_ROOT` | `data/dpo/runs` | Run output root |
| `DPO_REPORT_RUN` | `$(DPO_RUN)` | Run used by `dpo-report` |
| `DPO_INSPECT_RUN` | `$(DPO_REPORT_RUN)` | Run inspected by `dpo-inspect` |
| `DPO_MODEL` | `$(MODEL)` | Teacher model |
| `DPO_MAX_TOKENS` | `4096` | Teacher response max tokens |

## Reports

Smoke and target generation commands build their reports automatically. Rebuild reports manually when needed:

```bash
make pretrain-report PRETRAIN_REPORT_RUN=<run-id>
make distillation-sft-report DISTILLATION_SFT_REPORT_RUN=<run-id>
make sft-report SFT_REPORT_RUN=<run-id>
make dpo-report DPO_REPORT_RUN=<run-id>
```

## Test

```bash
make test
```

## Command Reference

See [docs/COMMANDS.md](docs/COMMANDS.md) for the full Make variable reference.
