import json
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import json
import pandas as pd
import re
import pymorphy3


# model_name = "finetuned_model2"
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Конфигурация для 4-бит квантования
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
# )

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# Загружаем основную модель для обучения
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
    device_map="cuda"
)

# Загружаем отдельную референсную модель и замораживаем её параметры
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
    device_map="cuda"
)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Определяем устройство
device = torch.device("cuda")
model.to(device)
ref_model.to(device)

special_words = open('матные_слова.txt', encoding='UTF-8',).read().split('\n')



def reward_function(
    responses,
    special_words,
    original_texts,
    length_weight=2.0, 
    offensive_weight=1.0,
    similarity_threshold=0.1  # порог близости длин
):
    morph = pymorphy3.MorphAnalyzer()
    
    # Вычисляем нормальные формы для специальных (запрещённых) слов.
    offensive_stems = [morph.parse(word)[0].normal_form for word in special_words]
    
    rewards = []
    for response, original in zip(responses, original_texts):
        response = re.sub(r'Перепиши кратко этот текст, используя мат: ', '', response.replace('\n', ''))
        lower_response = response.lower()
        tokens = lower_response.split()  # базовая токенизация
        
        offensive_count = 0
        for token in tokens:
            token_clean = re.sub(r'[^\w\s]', '', token)
            token_stem = morph.parse(token_clean)[0].normal_form
            if token_stem in offensive_stems:
                offensive_count += 1
        
        # Длина ответа и оригинального текста (в словах)
        response_length = len(tokens)
        original_length = len(original.split())
        
        # Базовая награда: если есть запрещённые слова, награда положительная,
        # иначе — фиксированный штраф.
        if offensive_count > 0:
            reward = (response_length * length_weight) + (offensive_count * offensive_weight)
        else:
            reward = -1.0
        
        # Если длина ответа превышает оригинал или почти совпадает с ней,
        # переопределяем награду как отрицательную.
        if original_length > 0:
            if response_length > original_length or abs(response_length - original_length) < similarity_threshold * original_length:
                reward = -abs(reward)
        
        
        print(
            f"Модель: {response}\n"
            f"Оригинал: {original}\n"
            f"Длина модели: {response_length}, Длина оригинала: {original_length}\n"
            f"Количество запрещенных слов: {offensive_count}\n"
            f"Reward: {reward}\n\n"
        )
        rewards.append(reward)
        
    return rewards


# Подготовка датасета с эффективной токенизацией
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


# Конфигурация PPO
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=32,
    mini_batch_size=8,
    gradient_accumulation_steps=4,
    ppo_epochs=4,
)

# Инициализация PPOTrainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collate_fn
)

device = torch.device("cuda")

# Параметры генерации
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128,
}

# Цикл обучения
for epoch in tqdm(range(ppo_config.ppo_epochs)):
    for i, batch in enumerate(ppo_trainer.dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # Добавляем корректные метки

        # Генерация ответов
        with torch.no_grad():
            response_tensors = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )

        # Декодирование ответов
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # Исходные тексты и эталонные ответы
        original_texts = batch["prompt_text"]
        reference_texts = batch["completion_text"]

        # Вычисление наград (ориентируемся на reference_texts)
        rewards_list = reward_function(responses, special_words, reference_texts)
        rewards = [torch.tensor(r, device=device, dtype=torch.float) for r in rewards_list]

        # Шаг PPO
        stats = ppo_trainer.step(
            [ids for ids in input_ids],
            [resp for resp in response_tensors],
            rewards
        )

        # Вывод лосса
        loss = stats.get("ppo_loss", None)
        if loss is not None:
            print(f"Epoch {epoch} - Iteration {i} - Loss: {loss}")

# Сохранение модели
model.save_pretrained("ppo_llama3")
tokenizer.save_pretrained("ppo_llama3")

