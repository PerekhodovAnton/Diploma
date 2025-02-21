import json
import torch
import re
import os
from tqdm import tqdm
import pymorphy3
from datasets import Dataset
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import PeftModel

# Пути к базовой модели и к LoRA адаптеру
base_model_path = "/home/jovyan/Effective_transmission_semantic_information/Diploma/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
base_model_path = "ai-forever/rugpt3small_based_on_gpt2"

# Загружаем токенизатор (лучше брать из базовой модели)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# Загружаем базовую модель с value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model_path,
    device_map="cuda"
)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model_path,
    device_map="cuda"
)
#ref_model = PeftModel.from_pretrained(base_ref_model, adapter_path)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

device = torch.device("cuda")
model.to(device)
ref_model.to(device)

# Загружаем список запрещённых слов
with open('матные_слова.txt', encoding='UTF-8') as f:
    special_words = f.read().splitlines()

def reward_function(
    responses,
    special_words,
    original_texts,
    length_weight=2.0, 
    offensive_weight=1.0,
    similarity_threshold=0.1
):
    morph = pymorphy3.MorphAnalyzer()
    offensive_stems = [morph.parse(word)[0].normal_form for word in special_words]
    
    rewards = []
    for response, original in zip(responses, original_texts):
        response = re.sub(r'Перепиши кратко этот текст, используя мат: ', '', response.replace('\n', ''))
        lower_response = response.lower()
        tokens = lower_response.split()
        
        offensive_count = 0
        for token in tokens:
            token_clean = re.sub(r'[^\w\s]', '', token)
            token_stem = morph.parse(token_clean)[0].normal_form
            if token_stem in offensive_stems:
                offensive_count += 1
        
        response_length = len(tokens)
        original_length = len(original.split())
        
        if offensive_count > 0:
            reward = (response_length * length_weight) + (offensive_count * offensive_weight)
        else:
            reward = -1.0
        
        if original_length > 0:
            if response_length > original_length or abs(response_length - original_length) < similarity_threshold * original_length:
                reward = -abs(reward)
        
        print(
            f"Модель: {response}\n"
            f"Оригинал: {original}\n"
            f"Длина модели: {response_length}, Длина оригинала: {original_length}\n"
            f"Количество запрещённых слов: {offensive_count}\n"
            f"Reward: {reward}\n\n"
            )
        rewards.append(reward)
    return rewards

def prepare_dataset(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        tokenized_prompt = tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        tokenized_completion = tokenizer(
            examples["completion"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized_prompt["input_ids"].squeeze(0),
            "attention_mask": tokenized_prompt["attention_mask"].squeeze(0),
            "labels": tokenized_completion["input_ids"].squeeze(0),
            "prompt_text": examples["prompt"],
            "completion_text": examples["completion"]
        }
    
    dataset = dataset.map(
        tokenize_function,
        batched=False,
        remove_columns=["prompt", "completion"],
        keep_in_memory=True
    )
    return dataset

# Загружаем датасет для PPO
dataset = prepare_dataset('ppo_diploma_dataset.json', tokenizer)

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]
    
    if isinstance(input_ids[0], list):
        input_ids = [torch.tensor(x) for x in input_ids]
        attention_mask = [torch.tensor(x) for x in attention_mask]
        labels = [torch.tensor(x) for x in labels]
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "prompt_text": [x["prompt_text"] for x in batch],
        "completion_text": [x["completion_text"] for x in batch]
            }

# # Конфигурация PPO (используем имя базовой модели для конфигурации)
# ppo_config = PPOConfig(
#     model_name=model,
#     learning_rate=1e-5,
#     batch_size=32,
#     mini_batch_size=8,
#     gradient_accumulation_steps=4,
#     ppo_epochs=4,
# )

# # Инициализируем PPOTrainer
# ppo_trainer = PPOTrainer(
#     config=ppo_config,
#     model=model,
#     ref_model=ref_model,
#     tokenizer=tokenizer,
#     dataset=dataset,
#     data_collator=collate_fn
# )

# generation_kwargs = {
#     "min_length": -1,
#     "top_k": 0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.eos_token_id,
#     "max_new_tokens": 128,
# }