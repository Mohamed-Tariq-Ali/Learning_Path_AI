"""
Enhanced Data Collection Script for Python Coding Assistant
Now includes the 18k Python instructions dataset from HuggingFace

Usage: python data/collect_data.py
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def create_custom_examples():
    """Create custom Python Q&A examples"""

    data = []

    # FastAPI specific questions
    fastapi_data = [
        {
            "text": """### Question: How do I add CORS middleware in FastAPI?
### Answer: Use CORSMiddleware to handle Cross-Origin requests:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "CORS enabled"}
```"""
        },
        {
            "text": """### Question: How do I use Pydantic models for request validation in FastAPI?
### Answer: Define Pydantic BaseModel classes for automatic validation:
```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., gt=0, lt=150)

app = FastAPI()

@app.post("/users/")
async def create_user(user: User):
    # FastAPI validates automatically
    return {"user": user, "message": "User created"}
```"""
        },
        {
            "text": """### Question: How do I handle file uploads in FastAPI?
### Answer: Use UploadFile for handling file uploads:
```python
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    # Save file
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(contents)

    return {
        "filename": file.filename,
        "size": len(contents),
        "content_type": file.content_type
    }
```"""
        },
        {
            "text": """### Question: How do I use dependency injection in FastAPI?
### Answer: FastAPI's dependency injection system manages shared logic:
```python
from fastapi import FastAPI, Depends, HTTPException

app = FastAPI()

# Simple dependency
def get_current_user(token: str):
    if not token:
        raise HTTPException(status_code=401)
    return {"user": "john", "token": token}

# Database dependency
def get_db():
    db = DatabaseConnection()
    try:
        yield db
    finally:
        db.close()

# Using dependencies
@app.get("/users/me")
async def read_users_me(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    return current_user
```"""
        }
    ]

    # Python fundamentals
    fundamentals_data = [
        {
            "text": """### Question: How do I use enumerate in Python?
### Answer: enumerate() adds a counter to an iterable:
```python
fruits = ['apple', 'banana', 'cherry']

# Basic usage
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# Start from custom index
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}. {fruit}")

# Create dictionary from enumerate
fruit_dict = {i: fruit for i, fruit in enumerate(fruits)}
```"""
        },
        {
            "text": """### Question: How do I work with sets in Python?
### Answer: Sets are unordered collections of unique elements:
```python
# Create sets
set1 = {1, 2, 3, 4, 5}
set2 = set([3, 4, 5, 6, 7])

# Set operations
union = set1 | set2          # {1, 2, 3, 4, 5, 6, 7}
intersection = set1 & set2   # {3, 4, 5}
difference = set1 - set2     # {1, 2}

# Add/remove elements
set1.add(6)
set1.remove(1)

# Check membership (O(1) time)
if 3 in set1:
    print("Found!")
```"""
        },
        {
            "text": """### Question: How do I use zip() in Python?
### Answer: zip() combines multiple iterables element-wise:
```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
cities = ['NY', 'LA', 'SF']

# Combine multiple lists
for name, age, city in zip(names, ages, cities):
    print(f"{name}, {age}, from {city}")

# Create dictionary
person_dict = dict(zip(names, ages))

# Unzip using zip with *
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
numbers, letters = zip(*pairs)
```"""
        }
    ]

    # Advanced Python
    advanced_data = [
        {
            "text": """### Question: How do I use lambda functions in Python?
### Answer: Lambda functions are anonymous single-expression functions:
```python
# Basic lambda
square = lambda x: x ** 2
print(square(5))  # 25

# With multiple arguments
add = lambda x, y: x + y

# In sorted()
students = [('Alice', 25), ('Bob', 20), ('Charlie', 23)]
sorted_students = sorted(students, key=lambda x: x[1])

# In map()
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))

# In filter()
evens = list(filter(lambda x: x % 2 == 0, numbers))
```"""
        },
        {
            "text": """### Question: How do I use the collections module in Python?
### Answer: collections provides specialized container datatypes:
```python
from collections import Counter, defaultdict, deque, namedtuple

# Counter - count occurrences
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counter = Counter(words)
print(counter.most_common(2))  # [('apple', 3), ('banana', 2)]

# defaultdict - default values for missing keys
dd = defaultdict(list)
dd['fruits'].append('apple')  # No KeyError

# deque - double-ended queue
dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)

# namedtuple - lightweight object
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
```"""
        }
    ]

    # Combine all custom data
    data.extend(fastapi_data)
    data.extend(fundamentals_data)
    data.extend(advanced_data)

    return data


def load_huggingface_dataset(max_samples=5000):
    """
    Load the 18k Python instructions dataset from HuggingFace
    Format it to match our training format
    """
    print(f"\n📡 Downloading HuggingFace dataset (iamtarun/python_code_instructions_18k_alpaca)...")
    print(f"   This may take a few minutes on first run...")

    try:
        # Load dataset
        dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

        print(f"   ✅ Downloaded {len(dataset)} examples")
        print(f"   📊 Using up to {max_samples} samples for training")

        # Limit samples if needed
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        formatted_data = []

        print(f"   🔄 Formatting data...")
        for item in tqdm(dataset, desc="   Processing"):
            # The dataset has 'instruction', 'input', and 'output' fields
            instruction = item.get('instruction', '').strip()
            input_text = item.get('input', '').strip()
            output = item.get('output', '').strip()

            # Skip empty entries
            if not instruction or not output:
                continue

            # Format the question
            if input_text:
                question = f"{instruction}\n{input_text}"
            else:
                question = instruction

            # Create training text in our format
            formatted_text = f"### Question: {question}\n### Answer: {output}"

            formatted_data.append({'text': formatted_text})

        print(f"   ✅ Formatted {len(formatted_data)} examples")
        return formatted_data

    except Exception as e:
        print(f"   ❌ Error loading HuggingFace dataset: {e}")
        print(f"   Continuing with custom examples only...")
        return []


def load_existing_data(filepath):
    """Load existing training data"""
    data = []
    if Path(filepath).exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data


def save_data(data, filepath):
    """Save data to JSONL format"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def deduplicate_data(data):
    """Remove duplicate entries based on text content"""
    seen = set()
    unique_data = []

    for item in data:
        text_hash = hash(item['text'][:200])  # Use first 200 chars for dedup
        if text_hash not in seen:
            seen.add(text_hash)
            unique_data.append(item)

    return unique_data


def main():
    print("🔍 Python Coding Assistant - Enhanced Data Collection")
    print("=" * 70)

    # Paths
    data_dir = Path(__file__).parent
    output_file = data_dir / "python_training_data.jsonl"

    all_data = []

    # 1. Load existing custom data
    print(f"\n📂 Step 1: Loading existing data from {output_file}...")
    existing_data = load_existing_data(output_file)
    if existing_data:
        print(f"   Found {len(existing_data)} existing examples")
        all_data.extend(existing_data)
    else:
        print(f"   No existing data found")

    # 2. Add custom examples
    print(f"\n🔨 Step 2: Generating custom examples...")
    custom_data = create_custom_examples()
    print(f"   Generated {len(custom_data)} custom examples")
    all_data.extend(custom_data)

    # 3. Load HuggingFace dataset
    print(f"\n📥 Step 3: Loading HuggingFace dataset...")
    hf_data = load_huggingface_dataset(max_samples=5000)  # Adjust max_samples as needed
    all_data.extend(hf_data)

    # 4. Deduplicate
    print(f"\n🔍 Step 4: Removing duplicates...")
    print(f"   Before deduplication: {len(all_data)} examples")
    unique_data = deduplicate_data(all_data)
    print(f"   After deduplication: {len(unique_data)} examples")
    print(f"   Removed {len(all_data) - len(unique_data)} duplicates")

    # 5. Save
    print(f"\n💾 Step 5: Saving to {output_file}...")
    save_data(unique_data, output_file)

    # 6. Statistics
    print(f"\n" + "=" * 70)
    print(f"✅ SUCCESS! Dataset ready for training")
    print(f"=" * 70)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total examples: {len(unique_data)}")
    print(f"   Average text length: {sum(len(d['text']) for d in unique_data) / len(unique_data):.0f} chars")
    print(f"   Shortest example: {min(len(d['text']) for d in unique_data)} chars")
    print(f"   Longest example: {max(len(d['text']) for d in unique_data)} chars")

    # Show sample
    print(f"\n📝 Sample training example:")
    print(f"   {unique_data[0]['text'][:300]}...")

    print(f"\n🚀 Next step: Run training with:")
    print(f"   python training/train_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

