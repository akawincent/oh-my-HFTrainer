# oh-my-HFTrainer

Minimal Hugging Face Trainer examples for fine-tuning `Qwen/Qwen3-0.6B` on
`karthiksagarn/astro_horoscope`.

The project keeps the Python path intentionally small: load a causal LM and
tokenizer, tokenize the dataset `horoscope` column, split train/test data, run
`transformers.Trainer`, evaluate perplexity, and save metrics and model outputs.
It also includes ready-to-edit examples for single-GPU training, FSDP,
DeepSpeed ZeRO-2, and Optuna hyperparameter search.

## Repository Layout

- `src/arguments.py`: dataclass-backed CLI arguments for model, data, training,
  HPO, and W&B options.
- `src/finetune_qwen3.py`: main training script. It saves eval metrics and the
  final model.
- `src/hpo_trial_qwen3.py`: lighter training/eval script used by HPO trials.
- `src/hpo_optuna_driver_qwen3.py`: Optuna driver that launches trial jobs with
  Accelerate and writes `best_hyperparameters.json`.
- `src/callbacks.py`: small Trainer callbacks for progress logging.
- `src/wandb_utils.py`: W&B environment setup when `--report_to wandb` is used.
- `scripts/finetune_qwen3.sh`: single-GPU entrypoint.
- `scripts/finetune_qwen3_fsdp.sh`: Accelerate FSDP entrypoint.
- `scripts/finetune_qwen3_deepspeed.sh`: Accelerate DeepSpeed ZeRO-2 entrypoint.
- `scripts/hpo_qwen3_ddp.sh`: Optuna HPO entrypoint using DDP trials.
- `configs/accelerate/`: DDP, FSDP, and DeepSpeed Accelerate configs.
- `configs/deepspeed/zero2.json`: DeepSpeed ZeRO stage 2 config.
- `release.py`: helper for uploading an output folder to Hugging Face Hub.

## Requirements

- Python `>=3.10`
- Linux GPU environment for actual fine-tuning
- CUDA-compatible PyTorch wheels from the configured `cu128` index
- `uv` for dependency management

Install dependencies from the repository root:

```bash
uv sync --extra dev
uv sync --extra finetune
```

`flash-attn` and `deepspeed` are CUDA-sensitive dependencies and may fail on
machines that do not match their build requirements.

## Quick Start

The default dataset is `karthiksagarn/astro_horoscope`. The preprocessing code
expects a `horoscope` text column in the dataset `train` split.

Run the single-GPU script:

```bash
export HF_HOME=/path/to/huggingface/cache
bash scripts/finetune_qwen3.sh
```

The single-GPU script currently pins `CUDA_VISIBLE_DEVICES=7`. Change that line
before running on a different machine or GPU layout.

## Distributed Runs

The checked-in Accelerate configs assume one machine with 8 processes.

Run FSDP:

```bash
export HF_HOME=/path/to/huggingface/cache
bash scripts/finetune_qwen3_fsdp.sh
```

Run DeepSpeed ZeRO-2:

```bash
export HF_HOME=/path/to/huggingface/cache
bash scripts/finetune_qwen3_deepspeed.sh
```

The DeepSpeed script currently captures a best-known HPO configuration in the
shell variables at the top of the file.

## Hyperparameter Search

Run Optuna HPO:

```bash
export HF_HOME=/path/to/huggingface/cache
bash scripts/hpo_qwen3_ddp.sh
```

The HPO driver launches each trial through Accelerate, reads the configured
metric from the trial output directory, and writes:

- `outputs/.../trial-*/`: per-trial metrics and artifacts
- `outputs/.../best_hyperparameters.json`: best objective value, params, and
  trial output directory

The default HPO objective is `eval_perplexity` with `minimize`.

## Weights And Biases

W&B is disabled unless Trainer reporting includes `wandb` or `all`.

Example flags:

```bash
--report_to wandb \
--wandb_project oh-my-hftrainer \
--wandb_mode online \
--wandb_watch false \
--wandb_log_model false
```

The distributed and HPO scripts show concrete W&B usage.

## Outputs

Training outputs are written under `outputs/` by default. This directory is for
local artifacts only and is ignored by git.

Upload a finished output directory to Hugging Face Hub:

```bash
./.venv/bin/python release.py outputs/Qwen3-0.6B-finetuned-astro-horoscope-fsdp \
  --repo-id your-namespace/your-model-name \
  --private
```

Use `--large-folder` for resumable whole-folder uploads when the output is large.

## Development

Prefer changing reusable defaults in `src/arguments.py` and keeping experiment
choices in `scripts/*.sh`.

Lint Python changes:

```bash
uv run ruff check src scripts
```

For lightweight training-flow changes, run at least an argument parse or help
check before launching a full job:

```bash
./.venv/bin/python src/finetune_qwen3.py --help
```

There is no formal test suite yet. For distributed or data-processing changes,
run a short smoke job before starting a full experiment.
