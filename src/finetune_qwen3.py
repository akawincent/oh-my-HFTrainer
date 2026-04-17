import math
import os

from datasets import load_dataset

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer

from arguments import DataArguments, ModelArguments, TrainingArguments, WandbArguments
from callbacks import EpochLoggerCallback, TrainLoggerCallback


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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        dtype=args.model_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer


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


def report_to_wandb(report_to):
    if isinstance(report_to, str):
        return report_to in {"all", "wandb"}

    return report_to is not None and ("all" in report_to or "wandb" in report_to)


def configure_wandb_environment(training_args, wandb_args):
    if not report_to_wandb(training_args.report_to):
        return

    if wandb_args.wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_args.wandb_project
    if wandb_args.wandb_mode:
        os.environ["WANDB_MODE"] = wandb_args.wandb_mode
    if wandb_args.wandb_watch:
        os.environ["WANDB_WATCH"] = wandb_args.wandb_watch
    if wandb_args.wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = wandb_args.wandb_log_model


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, WandbArguments)
    )
    model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

    configure_wandb_environment(training_args, wandb_args)

    # load model 
    model, tokenizer = load_qwen3_model(model_args)
    
    # process dataset
    dataset = process_dataset(tokenizer, data_args, model_args.model_max_length)

    # call Trainer
    trainer = CausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            EpochLoggerCallback(),
            TrainLoggerCallback()
        ]
    )
    trainer.train()

    # save model to disk
    trainer.save_model()
    
    # upload model to huggingface hub
    trainer.push_to_hub()
