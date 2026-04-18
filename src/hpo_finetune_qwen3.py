import json
import math
from pathlib import Path

from datasets import load_dataset

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer

from arguments import (
    DataArguments,
    HPOArguments,
    ModelArguments,
    TrainingArguments,
    WandbArguments
)
from callbacks import EpochLoggerCallback, TrainLoggerCallback
from wandb_utils import configure_wandb_environment


class CausalLMTrainer(Trainer):
    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
        output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        loss_key = f"{metric_key_prefix}_loss"
        if loss_key in output.metrics:
            try:
                output.metrics[f"{metric_key_prefix}_perplexity"] = math.exp(output.metrics[loss_key])
            except OverflowError:
                output.metrics[f"{metric_key_prefix}_perplexity"] = float("inf")
        return output


def load_qwen3_model(args):
    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        dtype=args.model_dtype,
    )


def load_qwen3_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_name_or_path)


def process_dataset(tokenizer, args, max_seq_len):
    def tokenize(batch):
        return tokenizer(
            batch["horoscope"],
            truncation=True,
            max_length=max_seq_len,
        )

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=args.test_split)

    return dataset


def save_best_hyperparameters(output_dir, best_run):
    output_path = Path(output_dir or "./outputs") / "best_hyperparameters.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": best_run.run_id,
        "objective": best_run.objective,
        "hyperparameters": best_run.hyperparameters,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved best hyperparameters to {output_path}")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (
            ModelArguments, 
            DataArguments, 
            TrainingArguments, 
            HPOArguments, 
            WandbArguments
        )
    )
    model_args, data_args, training_args, hpo_args, wandb_args = parser.parse_args_into_dataclasses()

    configure_wandb_environment(training_args, wandb_args)
    tokenizer = load_qwen3_tokenizer(model_args)
    dataset = process_dataset(tokenizer, data_args, model_args.model_max_length)

    def model_init(trial=None):
        return load_qwen3_model(model_args)

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                hpo_args.hpo_learning_rate_min,
                hpo_args.hpo_learning_rate_max,
                log=True,
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size",
                hpo_args.hpo_batch_size_choices,
            ),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps",
                hpo_args.hpo_gradient_accumulation_choices,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                hpo_args.hpo_weight_decay_min,
                hpo_args.hpo_weight_decay_max,
            ),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio",
                hpo_args.hpo_warmup_ratio_min,
                hpo_args.hpo_warmup_ratio_max,
            ),
        }

    def compute_objective(metrics):
        return metrics[hpo_args.hpo_metric]

    def trial_name(trial):
        trial_prefix = training_args.run_name or "optuna-hpo"
        return f"{trial_prefix}-trial-{trial.number:03d}"

    trainer = CausalLMTrainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            EpochLoggerCallback(),
            TrainLoggerCallback(),
        ],
    )
    best_run = trainer.hyperparameter_search(
        backend="optuna",
        direction=hpo_args.hpo_direction,
        hp_space=optuna_hp_space,
        n_trials=hpo_args.hpo_n_trials,
        compute_objective=compute_objective,
        hp_name=trial_name,
    )
    if training_args.process_index == 0 and best_run is not None:
        save_best_hyperparameters(training_args.output_dir, best_run)
