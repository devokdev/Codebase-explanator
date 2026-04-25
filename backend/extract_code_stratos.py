import json
from datasets import load_dataset

ds = load_dataset('bespokelabs/Bespoke-Stratos-17k')
train_split = ds['train']

print(f"Total rows in Bespoke-Stratos-17k: {len(train_split)}")

code_samples = []
keywords = ["python", "javascript", "java", "c++", "def ", "function ", "write a code", "algorithm"]

for i, item in enumerate(train_split):
    query = item['conversations'][0]['value']
    answer = item['conversations'][1]['value']
    
    # Check if query contains coding keywords
    if any(kw in query.lower() for kw in keywords):
        code_samples.append({
            "prompt": query,
            "thought": answer.split("<|begin_of_thought|>")[1].split("<|end_of_thought|>")[0].strip() if "<|begin_of_thought|>" in answer else "",
            "solution": answer.split("<|begin_of_solution|>")[1].split("<|end_of_solution|>")[0].strip() if "<|begin_of_solution|>" in answer else answer
        })
    
    if len(code_samples) >= 5:
        break

print(f"Found {len(code_samples)} preliminary code samples.")
with open("inspect_stratos_code.json", "w", encoding="utf-8") as f:
    json.dump(code_samples, f, indent=2, ensure_ascii=False)

print("Saved preliminary data to inspect_stratos_code.json")
