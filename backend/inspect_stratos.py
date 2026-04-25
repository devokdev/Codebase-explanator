import json
from datasets import load_dataset

ds = load_dataset('bespokelabs/Bespoke-Stratos-17k')
train_split = ds['train']

# Inspect the first 3 items
output_data = []
for i in range(3):
    output_data.append({
        "index": i,
        "conversations": train_split[i]['conversations']
    })

with open("inspect_stratos.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("Saved inspection data to inspect_stratos.json")
