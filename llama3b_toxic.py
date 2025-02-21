import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset

# Задаём имя модели и конфигурацию квантования (4bit)
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# quant_config = BitsAndBytesConfig(load_in_4bit=True)

# Загружаем модель с 4-bit квантованием и токенизатор
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quant_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загрузка датасета из JSON-файла (формат: список объектов с ключами "prompt" и "completion")
dataset = load_dataset("json", data_files="ru_paradetox_for_training.json")

# Функция для формирования обучающего примера в виде единой текстовой последовательности
def format_instruction(example):
    # Можно комбинировать prompt и completion через разделитель (например, "\nОтвет:")
    return {"text": f"{example['prompt']}\n{example['completion']}"}

# Преобразуем датасет в instruct-формат
dataset = dataset.map(format_instruction)

# Функция токенизации
def tokenize_function(examples):
    encodings = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    # Для обучения языковой модели labels равны input_ids
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

# Токенизируем датасет
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Определяем аргументы обучения
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Создаем Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Запускаем обучение
trainer.train()

# Сохраняем дообученную модель и токенизатор
trainer.save_model("./finetuned_llama")
tokenizer.save_pretrained("./finetuned_llama")
