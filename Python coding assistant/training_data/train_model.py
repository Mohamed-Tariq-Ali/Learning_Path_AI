"""
Train the Python Coding Assistant Model
Fine-tunes microsoft/phi-2 using LoRA + TRL SFTTrainer
OLD TRL VERSION COMPATIBLE
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer
from pathlib import Path

# ================= CONFIG =================
print("🚀 Python Coding Assistant - Training Script")
print("=" * 60)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "python_training_data.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "python-coding-assistant"

BASE_MODEL = "microsoft/phi-2"
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 2e-4

print(f"\n📁 Configuration:")
print(f"   Base Model: {BASE_MODEL}")
print(f"   Data File: {DATA_FILE}")
print(f"   Output Directory: {OUTPUT_DIR}")
print(f"   Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")

# ================= DATA CHECK =================
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Training data not found: {DATA_FILE}")

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n💻 Device: {device}")
if device == "cpu":
    print("⚠️ Training on CPU will be slow (expected)")

# ================= TOKENIZER =================
print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= MODEL =================
print(f"\n📥 Loading model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map="cpu",
)

model.config.use_cache = False
model.train()

print("✅ Model loaded")

# ================= LORA =================
print("\n⚙️ Configuring LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)

print("\n📊 Trainable parameters:")
model.print_trainable_parameters()

# ================= DATASET =================
print("\n📚 Loading dataset...")
dataset = load_dataset("json", data_files=str(DATA_FILE))["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(eval_dataset)}")

print("\nSample:")
print(train_dataset[0]["text"][:200])

# ================= TRAINING ARGS =================
print("\n🎯 Training configuration...")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    logging_steps=25,
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="none",
)

# ================= FORMATTING FUNC =================
def formatting_func(example):
    return example["text"]

# ================= TRAINER =================
print("\n👨‍🏫 Initializing SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
)

# ================= TRAIN =================
print("\n🏋️ Starting training...")
print("=" * 60)

trainer.train()

print("\n✅ Training completed")

# ================= SAVE =================
print("\n💾 Saving model...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("✅ Model saved")

# ================= TEST =================
print("\n🧪 Testing model...")

def test_model(prompt, max_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

tests = [
    "### Question: What is a list in Python?\n### Answer:",
    "### Question: Explain decorators.\n### Answer:",
]

for t in tests:
    print("\nQ:", t)
    print("A:", test_model(t)[:300])
    print("-" * 60)

print("\n🎉 TRAINING PIPELINE COMPLETE")
