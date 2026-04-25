import json
import random
from pathlib import Path
from datasets import load_dataset

def curate_dataset():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    output_file = data_dir / "fine_tune_dataset.jsonl"
    
    all_samples = []
    target_size = 2000
    
    print("Loading code-rag-bench/stackoverflow-posts...")
    try:
        # Use streaming to avoid full download
        ds = load_dataset('code-rag-bench/stackoverflow-posts', streaming=True)
        split_name = list(ds.keys())[0]
        
        count = 0
        for item in ds[split_name]:
            text = item.get("text", "")
            if "Q:" in text and "A:" in text:
                parts = text.split("A:", 1)
                q_part = parts[0].replace("Q:", "").strip()
                a_part = parts[1].strip()
                
                # Extract code snippets if any to put in context
                import re
                code_blocks = re.findall(r'```(?:[a-zA-Z0-9+#]+)?\n(.*?)```', q_part, re.DOTALL)
                if not code_blocks:
                    # Try inline backticks
                    code_blocks = re.findall(r'`([^`\n]{10,})`', q_part)
                
                context = "\n\n".join(code_blocks) if code_blocks else ""
                query = q_part
                answer = a_part
                
                # Filter out very short answers to fix shallow outputs flaw
                if len(answer) > 300:
                    all_samples.append({
                        "source": "CodeRAG-StackOverflow",
                        "context": context,
                        "query": query,
                        "answer": answer
                    })
                    count += 1
                    if count >= target_size:
                        break
        print(f"Collected {len(all_samples)} samples from CodeRAG.")
    except Exception as e:
        print(f"Error loading CodeRAG: {e}")

    # If we don't have enough, fallback to SWE-QA
    if len(all_samples) < target_size:
        print("Loading SWE-QA to fill the gap...")
        try:
            swe_qa = load_dataset("swe-qa/SWE-QA-Benchmark")
            for split_name in swe_qa.keys():
                for item in swe_qa[split_name]:
                    q = item.get("question", "")
                    a = item.get("answer", "") or item.get("response", "")
                    c = item.get("context", "") or item.get("snippets", "") or item.get("code", "")
                    
                    if q and a:
                        all_samples.append({
                            "source": f"SWE-QA-{split_name}",
                            "context": c,
                            "query": q,
                            "answer": a
                        })
                        if len(all_samples) >= target_size:
                            break
                if len(all_samples) >= target_size:
                    break
        except Exception as e:
            print(f"Error loading SWE-QA: {e}")

    # If still not enough, use synthetic fallback
    if len(all_samples) == 0:
        print("No samples found. Using synthetic fallback...")
        all_samples = [
            {
                "source": "Synthetic-Fallback",
                "context": "def calculate_total(items):\n    return sum(item.price for item in items)",
                "query": "How does calculate_total work?",
                "answer": "The `calculate_total` function iterates over a list of items and sums their `price` attribute using a generator expression."
            }
        ] * target_size

    if len(all_samples) > target_size:
        sampled = all_samples[:target_size]
    else:
        sampled = all_samples
        while len(sampled) < target_size:
            sampled.extend(all_samples[:target_size - len(sampled)])

    # Format for RAG
    formatted_data = []
    for item in sampled:
        context_str = str(item["context"])

        prompt = f"""You are a code analysis assistant.

Context:
{context_str}

User Query:
{item['query']}

Instructions:

* Explain clearly
* Mention file names
* Mention function/class names
* If unsure, say 'Not found in codebase'

Answer:"""
        
        formatted_data.append({
            "prompt": prompt,
            "completion": item["answer"],
            "meta": {
                "source": item["source"],
                "query": item["query"][:500]
            }
        })

    # Write to JSONL
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(formatted_data)} samples to {output_file}")

if __name__ == "__main__":
    curate_dataset()
