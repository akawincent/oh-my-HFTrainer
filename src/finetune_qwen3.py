from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

# load model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


# load and process datasets
def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")
dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)


# set training config
training_args = TrainingArguments(
    output_dir="./outputs/Qwen3-0.6B-finetuned-astro_horoscope",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    bf16=True,
    learning_rate=2e-5,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    dataloader_num_workers=8,         
    dataloader_pin_memory=True,
)

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
trainer.push_to_hub()
