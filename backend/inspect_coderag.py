import json
from datasets import load_dataset

try:
    # CodeRAG-Bench datastores are usually just a single split or have specific configs
    # Let's load stackoverflow-posts
    ds = load_dataset('code-rag-bench/stackoverflow-posts', streaming=True)
    
    # Get the first available split
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
        if count >= 3:
            break
            
    with open("inspect_coderag.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print("Saved inspection data to inspect_coderag.json")
except Exception as e:
    print(f"Error: {e}")
