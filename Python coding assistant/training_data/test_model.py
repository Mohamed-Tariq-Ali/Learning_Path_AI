"""
Test your trained Python Coding Assistant
Run this after training to verify the model works

Usage: python training/test_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "python-coding-assistant"
BASE_MODEL = "microsoft/phi-2"

print("🧪 Testing Python Coding Assistant")
print("=" * 60)

# Check if model exists
if not MODEL_PATH.exists():
    print(f"\n❌ Error: Model not found at {MODEL_PATH}")
    print(f"   Train the model first: python training/train_model.py")
    exit(1)

# Load model
print(f"\n📥 Loading model from {MODEL_PATH}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
model.eval()

print("   ✅ Model loaded successfully!\n")


def ask_question(question, max_tokens=200):
    """Ask the model a question"""
    prompt = f"### Question: {question}\n### Answer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    if "### Answer:" in response:
        answer = response.split("### Answer:")[-1].strip()
    else:
        answer = response.strip()

    return answer


# Test questions
test_cases = [
    "How do I create a virtual environment in Python?",
    "What's the difference between a list and a tuple?",
    "How do I use async/await in FastAPI?",
    "How do I read a JSON file?",
    "What are Python decorators?",
]

print("=" * 60)
print("🔍 RUNNING TEST QUESTIONS")
print("=" * 60)

for i, question in enumerate(test_cases, 1):
    print(f"\n📝 Test {i}/{len(test_cases)}")
    print(f"Q: {question}")
    print("-" * 60)

    answer = ask_question(question, max_tokens=250)
    print(f"A: {answer}")
    print("=" * 60)

# Interactive mode
print("\n\n💬 Interactive Mode (type 'quit' to exit)")
print("=" * 60)

while True:
    try:
        user_question = input("\n🤔 Your question: ").strip()

        if user_question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not user_question:
            continue

        print("\n🤖 Answer:")
        answer = ask_question(user_question, max_tokens=300)
        print(answer)
        print("-" * 60)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")