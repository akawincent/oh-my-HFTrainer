#!/usr/bin/env bash

set -euo pipefail

# Model configuration
llm=Qwen/Qwen3-0.6B
attn_implementation=sdpa
max_token_length=256

# Training hyperparameters
optim=adamw_torch
lr=0.0008028885185834784
weight_decay=0.042141114286418216
warmup_ratio=0.018679858620030212
# Best HPO setting: 8 GPUs * 16 per device * 1 grad accumulation = 128 global batch.
batch_size=16
grad_accum_steps=1
max_grad_norm=1.0
epochs=5
grad_ckpt_flag=False

# Logging strategy
eval_strategy=epoch
save_strategy=best
logging_steps=50
load_best_model_flag=True
metric=perplexity
greater_is_better_flag=False
run_name=qwen3-hpo-horoscope-best

# Dataloader config
num_workers=8
pin_memory_flag=True

# Weights & Biases configuration
wandb_project=hpo_a800x8_ddp
wandb_mode=online
wandb_watch=false
wandb_log_model=false

# Training entry point
entry_file=src/finetune_qwen3.py

# Dataset configuration
datasets=karthiksagarn/astro_horoscope

# Output configuration
output_dir=outputs/Qwen3-0.6B-finetuned-astro-horoscope-deepspeed-best-hpo

args=(
    --model_name_or_path "${llm}"
    --model_max_length "${max_token_length}"
    --attn_implementation "${attn_implementation}"
    --dataset_name "${datasets}"
    --output_dir "${output_dir}"
    --bf16 True
    --num_train_epochs "${epochs}"
    --per_device_train_batch_size "${batch_size}"
    --gradient_accumulation_steps "${grad_accum_steps}"
    --gradient_checkpointing "${grad_ckpt_flag}"
    --optim "${optim}"
    --learning_rate "${lr}"
    --weight_decay "${weight_decay}"
    --warmup_ratio "${warmup_ratio}"
    --max_grad_norm "${max_grad_norm}"
    --deepspeed configs/deepspeed/zero2.json
    --metric_for_best_model "${metric}"
    --greater_is_better "${greater_is_better_flag}"
    --eval_strategy "${eval_strategy}"
    --save_strategy "${save_strategy}"
    --logging_steps "${logging_steps}"
    --load_best_model_at_end "${load_best_model_flag}"
    --report_to wandb
    --run_name "${run_name}"
    --wandb_project "${wandb_project}"
    --wandb_mode "${wandb_mode}"
    --wandb_watch "${wandb_watch}"
    --wandb_log_model "${wandb_log_model}"
    --dataloader_num_workers "${num_workers}"
    --dataloader_pin_memory "${pin_memory_flag}"
)

./.venv/bin/accelerate launch \
    --config_file configs/accelerate/ds_config.yaml \
    "${entry_file}" "${args[@]}"
