import json
import torch, gc
gc.collect()
torch.cuda.empty_cache()
import re
import os
os.environ['HF_HOME'] = '/home/jovyan/Effective_transmission_semantic_information/Diploma'
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

model_name = "/home/jovyan/Effective_transmission_semantic_information/Diploma/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    attn_implementation='eager'
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="ru_paradetox_for_training.json")['train']

def format_instruction(example):
    return {"text": f"{example['prompt']} Ответ: {example['completion']}"}

dataset = dataset.map(format_instruction)

def tokenize_function(examples):
    encodings = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize_function, batched=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none'
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./finetuned_gemma",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Сохраняем дообученную модель и токенизатор
trainer.save_model("./finetuned_gemma")
tokenizer.save_pretrained("./finetuned_gemma")
