import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

base_dir = Path(__file__).resolve().parent.parent
input_file = base_dir / "data" / "fine_tune_dataset.jsonl"
output_file = base_dir / "data" / "fine_tune_dataset_enriched.jsonl"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env", flush=True)
    exit(1)

client = genai.Client(api_key=api_key)

def enrich_sample(sample):
    query = sample["meta"]["query"]
    shallow_answer = sample["completion"]
    
    prompt = f"""You are an expert AI software engineer and code analysis teacher.
I have a real-world developer question about a codebase, and a shallow answer.
Since I don't have the original code snippet, I need you to:
1. Synthesize a realistic, syntactically valid, and highly relevant Python or JavaScript code snippet that would serve as the context for this question.
2. Generate a DEEP, REASONING-RICH explanation that answers the question step-by-step.

Your response MUST follow this exact format:
---CODE_START---
[Insert realistic code snippet here]
---CODE_END---
---ANSWER_START---
### Step-by-Step Breakdown
[Detailed logic flow]

### Variable & Control Flow Analysis
[Breakdown of key components]

### Architectural Insight
[Deep explanation]
---ANSWER_END---

Question: {query}
Shallow Answer: {shallow_answer}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
            )
            text = response.text
            
            if "---CODE_START---" in text and "---CODE_END---" in text and "---ANSWER_START---" in text and "---ANSWER_END---" in text:
                code_part = text.split("---CODE_START---")[1].split("---CODE_END---")[0].strip()
                answer_part = text.split("---ANSWER_START---")[1].split("---ANSWER_END---")[0].strip()
                
                final_prompt = f"""You are a code analysis assistant.

Context:
{code_part}

User Query:
{query}

Instructions:

* Explain clearly
* Mention file names
* Mention function/class names
* If unsure, say 'Not found in codebase'

Answer:"""
                
                return {
                    "prompt": final_prompt,
                    "completion": answer_part,
                    "meta": sample["meta"]
                }
        except errors.APIError as e:
            if "429" in str(e):
                print(f"Rate limited (429). Sleeping 30 seconds...", flush=True)
                time.sleep(30)
            else:
                print(f"API Error: {e}", flush=True)
                time.sleep(5)
        except Exception as e:
            print(f"Error: {e}", flush=True)
            time.sleep(2)
            
    return None

def main():
    print("Loading shallow dataset...", flush=True)
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            
    processed_queries = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_queries.add(item["meta"]["query"])
                except:
                    pass
    
    to_process = [s for s in samples if s["meta"]["query"] not in processed_queries]
    
    target_samples = 100
    current_count = len(processed_queries)
    
    if current_count >= target_samples:
        print(f"Already have {current_count} samples. Goal met!", flush=True)
        return
        
    needed = target_samples - current_count
    to_process = to_process[:needed]
    print(f"Processing sequentially: {len(to_process)} samples needed.", flush=True)
    
    with open(output_file, "a", encoding="utf-8") as f:
        for count, sample in enumerate(to_process, start=1):
            print(f"Enriching sample {current_count + count}/{target_samples}...", flush=True)
            result = enrich_sample(sample)
            if result:
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(f"Saved enriched sample.", flush=True)
            else:
                print(f"Failed to enrich sample.", flush=True)
            
            time.sleep(5)

if __name__ == "__main__":
    main()
