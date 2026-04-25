import json
from datasets import load_dataset

try:
    ds = load_dataset('nvidia/OpenCodeReasoning', 'split_0')
    train_split = ds['train']
    
    output_data = []
    # Inspect the first 3 items
    for i in range(min(3, len(train_split))):
        output_data.append({
            "index": i,
            "columns": list(train_split[i].keys()),
            "sample": {k: str(train_split[i][k])[:500] for k in train_split[i].keys()}
        })
        
    with open("inspect_nvidia.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print("Saved inspection data to inspect_nvidia.json")
except Exception as e:
    print(f"Error: {e}")
