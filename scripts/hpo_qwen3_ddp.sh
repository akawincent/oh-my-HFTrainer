#!/usr/bin/env bash
set -euo pipefail

# Model configuration
llm=Qwen/Qwen3-0.6B
attn_implementation=sdpa
max_token_length=256

# Dataset configuration
datasets=karthiksagarn/astro_horoscope

# HPO search configuration
trials=12
epochs=3
logging_steps=10
hpo_metric=eval_perplexity
hpo_direction=minimize
run_name=qwen3-hpo-horoscope

# Weights & Biases configuration
wandb_project=hpo_a800x8_ddp
wandb_mode=online

# Search space
lr_min=5e-6
lr_max=1e-3
batch_size_choices=(16 32)
grad_accum_choices=(1 2 4 8)
weight_decay_min=0.0
weight_decay_max=0.1
warmup_ratio_min=0.0
warmup_ratio_max=0.1

# Dataloader config
num_workers=8
pin_memory_flag=True

# Training entry point
entry_file=src/hpo_optuna_driver_qwen3.py

# Output configuration
output_dir=./outputs/Qwen3-0.6B-hpo-horoscope-wandb

args=(
    --model_name_or_path "${llm}"
    --model_max_length "${max_token_length}"
    --attn_implementation "${attn_implementation}"
    --dataset_name "${datasets}"
    --output_dir "${output_dir}"
    --hpo_n_trials "${trials}"
    --hpo_metric "${hpo_metric}"
    --hpo_direction "${hpo_direction}"
    --hpo_accelerate_config configs/accelerate/ddp_config.yaml
    --hpo_train_entry_file src/hpo_trial_qwen3.py
    --hpo_learning_rate_min "${lr_min}"
    --hpo_learning_rate_max "${lr_max}"
    --hpo_batch_size_choices "${batch_size_choices[@]}"
    --hpo_gradient_accumulation_choices "${grad_accum_choices[@]}"
    --hpo_weight_decay_min "${weight_decay_min}"
    --hpo_weight_decay_max "${weight_decay_max}"
    --hpo_warmup_ratio_min "${warmup_ratio_min}"
    --hpo_warmup_ratio_max "${warmup_ratio_max}"
    --bf16 True
    --num_train_epochs "${epochs}"
    --eval_strategy epoch
    --save_strategy no
    --logging_steps "${logging_steps}"
    --load_best_model_at_end False
    --metric_for_best_model "${hpo_metric}"
    --greater_is_better False
    --dataloader_num_workers "${num_workers}"
    --dataloader_pin_memory "${pin_memory_flag}"
    --report_to wandb
    --run_name "${run_name}"
    --wandb_project "${wandb_project}"
    --wandb_mode "${wandb_mode}"
)

./.venv/bin/python "${entry_file}" "${args[@]}"
