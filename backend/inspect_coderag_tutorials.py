import json
from datasets import load_dataset

try:
    ds = load_dataset('code-rag-bench/online-tutorials', streaming=True)
    split_name = list(ds.keys())[0]
    print(f"Loaded splits: {list(ds.keys())}")
    
    output_data = []
    count = 0
    for item in ds[split_name]:
        output_data.append({
            "index": count,
            "keys": list(item.keys()),
            "sample": {k: str(item[k])[:500] for k in item.keys()}
        })
        count += 1
        if count >= 2:
            break
            
    with open("inspect_coderag_tutorials.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print("Saved inspection data to inspect_coderag_tutorials.json")
except Exception as e:
    print(f"Error: {e}")
