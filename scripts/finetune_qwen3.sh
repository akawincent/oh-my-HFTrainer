# Model configuration
llm=Qwen/Qwen3-0.6B  # Using HuggingFace model ID
attn_implementation=sdpa

# Training hyperparameters
optim=adamw_torch
lr=2e-4
batch_size=64
grad_accum_steps=1
epochs=3
grad_ckpt_flag=False

# Logging strategy
eval_strategy=epoch
save_strategy=best
logging_steps=50

# Dataloader config
num_workers=8
pin_memory_flag=True 

# Training entry point
entry_file=src/finetune_qwen3.py

# Dataset configuration 
datasets=karthiksagarn/astro_horoscope

# Output configuration
output_dir=./outputs/Qwen3-0.6B-finetuned-astro_horoscope_use_sdpa

# Training arguments
args="
    --model_name_or_path "${llm}" \
    --model_max_length 512 \
    --attn_implementation "${attn_implementation}" \
    --dataset_name ${datasets} \
    --output_dir ${output_dir} \
    --bf16 True \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --gradient_checkpointing ${grad_ckpt_flag} \
    --optim "${optim}" \
    --learning_rate ${lr} \
    --eval_strategy ${eval_strategy} \
    --save_strategy ${save_strategy} \
    --logging_steps ${logging_steps} \
    --dataloader_num_workers ${num_workers} \
    --dataloader_pin_memory ${pin_memory_flag} \
"

CUDA_VISIBLE_DEVICES=7 python ${entry_file} ${args}