import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = 'ppo_llama3'


# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# Загружаем основную модель для обучения
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda"
)
prompt = input()
def generate_response(model, text, tokenizer, max_length=100):
    prompt = input()
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=1.0,
            top_k=0.0,
            top_p=0.95,
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)



model = generate_response(model_name, tokenizer, prompt)
print(model)