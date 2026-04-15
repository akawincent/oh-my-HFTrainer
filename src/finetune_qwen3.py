from datasets import load_dataset

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer

from arguments import ModelArguments, DataArguments, TrainingArguments


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


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load model 
    model, tokenizer = load_qwen3_model(model_args)
    
    # process dataset
    dataset = process_dataset(tokenizer, data_args, model_args.model_max_length)

    # call Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # upload model to huggingface hub
    # trainer.push_to_hub()
