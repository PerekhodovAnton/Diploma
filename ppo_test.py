import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'finetuned_llama'

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Загружаем модель
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda"
)



def generate_response(model, text, tokenizer, max_length=256):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt")
    # Переносим все тензоры в устройство модели
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=1.0,
            top_k=0,
            top_p=0.95,
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

while True:
    prompt = input("введите запрос:")
    response = generate_response(model, prompt, tokenizer)
    print(response)
