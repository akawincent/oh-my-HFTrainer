#!/usr/bin/env bash

set -euo pipefail

export HF_HOME=/home/dongchengrui/.cache/huggingface

# Model configuration
llm=Qwen/Qwen3-0.6B
attn_implementation=sdpa
max_token_length=512

# Training hyperparameters
optim=adamw_torch
lr=2e-4
batch_size=64
grad_accum_steps=1
epochs=5
grad_ckpt_flag=False

# Logging strategy
eval_strategy=epoch
save_strategy=best
logging_steps=50
load_best_model_flag=True
metric=perplexity
greater_is_better_flag=False

# Dataloader config
num_workers=8
pin_memory_flag=True

# Training entry point
entry_file=src/finetune_qwen3.py

# Dataset configuration
datasets=karthiksagarn/astro_horoscope

# Output configuration
output_dir=outputs/Qwen3-0.6B-finetuned-astro-horoscope-fsdp

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
    --metric_for_best_model "${metric}"
    --greater_is_better "${greater_is_better_flag}"
    --eval_strategy "${eval_strategy}"
    --save_strategy "${save_strategy}"
    --logging_steps "${logging_steps}"
    --load_best_model_at_end "${load_best_model_flag}"
    --dataloader_num_workers "${num_workers}"
    --dataloader_pin_memory "${pin_memory_flag}"
)

accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    ${entry_file} "${args[@]}"
