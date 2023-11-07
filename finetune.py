import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load Model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float32)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Post processing

print(model)

for param in model.parameters():
    param.requires_grad = False

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

model.lm_head = CastOutputToFloat(model.lm_head)

# LoRA

config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.to(device)

# Train

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=15,
        learning_rate=3e-4,
        logging_steps=1,
        output_dir="outputs",
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

# Inference
while True:
    line = input('Enter a quote to complete: ')
    batch = tokenizer(line, return_tensors="pt")
    batch.to(device)

    model.eval()
    output_tokens = model.generate(**batch, max_new_tokens=50)

    print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
