import json
from datasets import load_dataset

try:
    ds = load_dataset('nvidia/OpenCodeReasoning', 'split_0', streaming=True)
    split_data = ds['split_0']
    
    for item in split_data:
        # Print the keys and the first 500 chars of values
        print(f"Keys: {list(item.keys())}")
        for k, v in item.items():
            print(f"--- {k} ---")
            print(str(v)[:500])
        break
except Exception as e:
    print(f"Error: {e}")
