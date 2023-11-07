import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float32)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

while True:
    line = input('Enter a quote to complete: ')
    batch = tokenizer(line, return_tensors="pt")
    batch.to(device)

    model.eval()
    output_tokens = model.generate(**batch, max_new_tokens=50)

    print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
