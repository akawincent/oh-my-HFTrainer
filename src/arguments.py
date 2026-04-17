import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-0.6B")
    model_dtype: Optional[str] = field(default="auto")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    attn_implementation: Optional[str] = field(default="eager")


@dataclass
class DataArguments:
    dataset_name: str = field(default="karthiksagarn/astro_horoscope")
    test_split: float = field(default=0.1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default=None)

    dataloader_num_workers: int = field(default=8)
    dataloader_pin_memory: bool = field(default=True)

    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=True)
    learning_rate: float = field(default=1e-5)

    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)
    gradient_checkpointing: bool = field(default=False)

    report_to: None | str | list[str] = field(
        default="none",
        metadata={
            "help": (
                "The list of integrations to report the results and logs to. "
                "Use 'all' for all installed integrations, 'none' for no integrations."
            )
        },
    )
    run_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "An optional descriptor for the run. Notably used for trackio, wandb, "
                "mlflow comet and swanlab logging."
            )
        },
    )
    logging_steps: int = field(default=100)
    eval_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: Optional[str] = field(default="loss")
    greater_is_better: bool = field(default=True)


@dataclass
class HPOArguments:
    hpo_n_trials: int = field(default=12)
    hpo_direction: str = field(default="minimize")
    hpo_metric: str = field(default="eval_perplexity")
    hpo_learning_rate_min: float = field(default=5e-6)
    hpo_learning_rate_max: float = field(default=5e-4)
    hpo_batch_size_choices: list[int] = field(default_factory=lambda: [16, 32, 64])
    hpo_gradient_accumulation_choices: list[int] = field(default_factory=lambda: [1, 2, 4])
    hpo_weight_decay_min: float = field(default=0.0)
    hpo_weight_decay_max: float = field(default=0.1)
    hpo_warmup_ratio_min: float = field(default=0.0)
    hpo_warmup_ratio_max: float = field(default=0.1)


@dataclass
class WandbArguments:
    wandb_project: Optional[str] = field(default="oh-my-hftrainer")
    wandb_mode: Optional[str] = field(default=None)
    wandb_watch: Optional[str] = field(default="false")
    wandb_log_model: Optional[str] = field(default="false")
