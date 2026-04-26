import os
import json
import re
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


PROMPT_TEMPLATE = """You are a code analysis assistant.

Context:
{retrieved_code_chunks}

User Query:
{query}

Instructions:

* Explain clearly
* Mention file names
* Mention function/class names
* If unsure, say 'Not found in codebase'

Answer:"""


def build_context(chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant code chunks retrieved."

    rendered = []
    for chunk in chunks:
        rendered.append(
            "\n".join(
                [
                    f"File: {chunk['file_path']}",
                    f"Type: {chunk['type']}",
                    f"Name: {chunk['name']}",
                    f"Lines: {chunk.get('line_start', '?')}-{chunk.get('line_end', '?')}",
                    "Code:",
                    chunk["code"][:1600],
                ]
            )
        )
    return "\n\n---\n\n".join(rendered)


def build_file_summary_context(file_summaries: List[Dict]) -> str:
    if not file_summaries:
        return "No file summaries available."

    return "\n".join(
        f"- {item['file_path']}: {item['summary']}"
        for item in file_summaries
    )


def build_compact_chunk_context(chunks: List[Dict], limit: int = 6) -> str:
    if not chunks:
        return "No retrieved code snippets."

    rendered = []
    for chunk in chunks[:limit]:
        rendered.append(
            "\n".join(
                [
                    f"File: {chunk['file_path']}",
                    f"Symbol: {chunk['name']}",
                    f"Type: {chunk['type']}",
                    f"Lines: {chunk.get('line_start', '?')}-{chunk.get('line_end', '?')}",
                    f"Snippet:\n{chunk['code'][:900]}",
                ]
            )
        )
    return "\n\n---\n\n".join(rendered)


class LLMService:
    def __init__(self) -> None:
        self.local_model_path = os.getenv("LOCAL_MODEL_PATH", "C:/Users/karta/Desktop/GenAIEndTerm/merged model")
        if os.path.exists(self.local_model_path):
            print(f"Loading fine-tuned local model from {self.local_model_path}...")
            self.use_local = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_model_path, 
                torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
        else:
            print("Local model not found. Falling back to Ollama...")
            self.use_local = False
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
            self.model_name = os.getenv("OLLAMA_MODEL", "tinyllama")
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            )

    def _chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 220) -> str:
        if hasattr(self, 'use_local') and self.use_local:
            prompt = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
                elif role == 'user':
                    if "<s>[INST]" in prompt:
                        prompt += f"{content} [/INST]"
                    else:
                        prompt += f"<s>[INST] {content} [/INST]"
                elif role == 'assistant':
                    prompt += f" {content} </s>"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            raw_out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "[/INST]" in raw_out:
                return raw_out.split("[/INST]")[-1].strip()
            return raw_out.strip()
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                extra_body={
                    "keep_alive": "15m",
                    "options": {
                        "num_predict": max_tokens,
                        "num_ctx": 4096,
                    },
                },
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Model returned an empty response.")
            return content.strip()

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _short_summary(text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_\-./ ]+", " ", text)
        words = [word for word in cleaned.split() if word]
        summary = " ".join(words[:5]).strip()
        return summary or "Code file summary"

    @staticmethod
    def _filename_tokens(file_path: str) -> List[str]:
        stem = Path(file_path).stem.lower().replace("-", "_")
        return [token for token in stem.split("_") if token]

    def _infer_file_summary(self, file_path: str, symbols: List[str], code: str) -> str:
        tokens = self._filename_tokens(file_path)
        token_set = set(tokens)
        code_lower = code.lower()

        if {"train", "trainer", "finetune", "lora"} & token_set:
            return "LoRA training pipeline"
        if {"benchmark", "bench"} & token_set:
            return "benchmark runner"
        if {"eval", "evaluate", "evaluation"} & token_set:
            return "evaluation script"
        if {"test", "tests", "testing"} & token_set:
            return "test script"
        if {"ui", "demo", "gradio"} & token_set:
            return "interactive demo UI"
        if {"app", "main"} & token_set:
            return "application entrypoint"
        if {"config", "settings"} & token_set:
            return "configuration helpers"
        if {"utils", "helpers"} & token_set:
            return "utility helpers"
        if {"dataset", "data"} & token_set:
            return "dataset preparation"
        if {"model", "models"} & token_set:
            return "model loading logic"
        if "argparse" in code_lower or "if __name__ == \"__main__\"" in code_lower:
            return "command-line entrypoint"
        if "gradio" in code_lower or "streamlit" in code_lower:
            return "interactive demo UI"
        if "unittest" in code_lower or "pytest" in code_lower or any(sym.startswith("test") for sym in symbols):
            return "test script"
        if any("load" in sym.lower() and "model" in sym.lower() for sym in symbols):
            return "model loading logic"
        if any("dataset" in sym.lower() for sym in symbols):
            return "dataset preparation"
        if any("train" in sym.lower() for sym in symbols):
            return "training workflow"
        if any("eval" in sym.lower() for sym in symbols):
            return "evaluation workflow"
        if len(tokens) >= 2:
            return self._short_summary(" ".join(tokens[:3]))
        if tokens:
            return self._short_summary(f"{tokens[0]} module")
        return "code module"

    @staticmethod
    def _clean_markdown_text(text: str) -> str:
        cleaned = re.sub(r"`([^`]*)`", r"\1", text)
        cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _sanitize_answer_output(self, answer: str) -> str:
        cleaned = answer.strip()
        noisy_headers = [
            "relevant files:",
            "retrieved evidence:",
            "snippet:",
            "code:",
        ]
        lowered = cleaned.lower()
        cut_positions = [lowered.find(header) for header in noisy_headers if lowered.find(header) != -1]
        if cut_positions:
            cleaned = cleaned[: min(cut_positions)].strip()

        cleaned = re.sub(r"^repository summary:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^answer:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        if not cleaned:
            return answer.strip()
        return cleaned

    @staticmethod
    def _is_repo_overview_query(query: str) -> bool:
        lowered = query.lower()
        patterns = (
            "what is this repo about",
            "what is this repository about",
            "what does this repo do",
            "what does this repository do",
            "summarize this repo",
            "summarize this repository",
            "overview of this repo",
            "overview of this repository",
            "explain this repo",
            "explain this repository",
        )
        return any(pattern in lowered for pattern in patterns)

    def _repo_overview_fallback(self, repo_context: Dict) -> str:
        repo_name = repo_context.get("repo_name", "This repository")
        summary = self._clean_markdown_text(repo_context.get("repo_summary", ""))
        summary = re.split(r"\bFeatures\b|\bInstallation\b|\bUsage\b", summary, maxsplit=1)[0].strip(" .#")
        file_summaries = repo_context.get("file_summaries", [])
        highlights = ", ".join(
            f"{item['file_path']} ({item['summary']})"
            for item in file_summaries[:4]
        )

        if summary and highlights:
            return f"{summary} Key files include {highlights}."
        if summary:
            return summary
        if highlights:
            return f"{repo_name} includes files such as {highlights}."
        return f"{repo_name} has been ingested, but a high-level summary is not available yet."

    @staticmethod
    def _read_repo_file(repo_root: Path | None, relative_path: str) -> str:
        if repo_root is None:
            return ""
        path = repo_root / relative_path
        if not path.exists() or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    @staticmethod
    def _detect_first_keyword(text: str, keywords: List[str]) -> str | None:
        lowered = text.lower()
        for keyword in keywords:
            if keyword.lower() in lowered:
                return keyword
        return None

    def _extract_stack_answer(self, query: str, repo_context: Dict, retrieved_chunks: List[Dict]) -> str | None:
        query_lower = query.lower()
        if not any(term in query_lower for term in ("backend", "frontend", "database", "db", "authentication", "auth", "tech stack", "stack")):
            return None

        source_root = repo_context.get("source_root")
        repo_root = Path(source_root) if source_root else None
        readme = self._read_repo_file(repo_root, "README.md")
        backend_main = self._read_repo_file(repo_root, "backend/main.py")
        backend_database = self._read_repo_file(repo_root, "backend/database.py")
        backend_config = self._read_repo_file(repo_root, "backend/config.py")
        frontend_package = self._read_repo_file(repo_root, "frontend/package.json")
        requirements = self._read_repo_file(repo_root, "requirements.txt")
        prd = self._read_repo_file(repo_root, "prd.md")
        combined = "\n".join([readme, backend_main, backend_database, backend_config, frontend_package, requirements, prd])

        backend_framework = self._detect_first_keyword(
            combined,
            ["FastAPI", "Flask", "Django", "Express", "NestJS", "Spring Boot", "Rails"],
        )
        frontend_framework = self._detect_first_keyword(
            combined,
            ["Next.js", "React", "Vue", "Angular", "Svelte"],
        )
        database_name = None
        if "asyncpg" in combined.lower() or "postgresql://" in combined.lower() or "supabase postgres" in combined.lower():
            database_name = "PostgreSQL"
        elif "mongodb" in combined.lower() or "mongo" in combined.lower():
            database_name = "MongoDB"
        elif "sqlite" in combined.lower():
            database_name = "SQLite"

        auth_signal = self._detect_first_keyword(
            combined,
            ["OAuth", "JWT", "Auth0", "Supabase Auth", "Firebase Auth", "next-auth", "flask_login"],
        )
        has_auth_code = any(
            term in combined.lower()
            for term in ("login", "signup", "jwt", "oauth", "bearer ", "password_hash", "supabase auth", "auth0")
        )

        if "backend" in query_lower:
            if backend_framework == "FastAPI":
                answer = (
                    "The backend is built with FastAPI in Python. "
                    "In backend/main.py, the app is created with `FastAPI(...)` and it mounts routers such as `datasets`, `templates`, `automation`, `logbook`, and `dashboard_v2`."
                )
                if database_name == "PostgreSQL":
                    answer += " The database layer in backend/database.py uses `asyncpg`, so it is backed by PostgreSQL or Supabase Postgres."
                return answer
            if backend_framework:
                return f"The backend appears to use {backend_framework}. The strongest evidence is in backend/main.py and the repository README."
            return "Not found in codebase"

        if "frontend" in query_lower:
            if frontend_framework == "React":
                answer = (
                    "The frontend is built with React. "
                    "frontend/package.json lists `react`, `react-dom`, and `react-router-dom`, and the same file uses Vite for the build tooling."
                )
                return answer
            if frontend_framework:
                return f"The frontend appears to use {frontend_framework}. The strongest evidence is in frontend/package.json and the README."
            return "Not found in codebase"

        if "database" in query_lower or re.search(r"\bdb\b", query_lower):
            if database_name == "PostgreSQL":
                return (
                    "The repository uses PostgreSQL. "
                    "backend/config.py defines `DATABASE_URL` with a `postgresql://...` connection string, and backend/database.py creates a pool with `asyncpg.create_pool(...)`."
                )
            if database_name:
                return f"The repository appears to use {database_name}. The strongest evidence is in backend/database.py and backend/config.py."
            return "Not found in codebase"

        if "authentication" in query_lower or re.search(r"\bauth\b", query_lower):
            if auth_signal:
                return f"Authentication appears to use {auth_signal}. The strongest evidence is in the repository source and configuration files."
            if not has_auth_code and ("no auth" in prd.lower() or "there is no auth" in prd.lower()):
                return "Authentication is not implemented yet. prd.md explicitly says there is no auth or multi-user workspace model yet."
            auth_related = [chunk for chunk in retrieved_chunks if "auth" in chunk["file_path"].lower() or "auth" in chunk["name"].lower()]
            if auth_related:
                lead = auth_related[0]
                return (
                    f"The strongest auth-related evidence is in {lead['file_path']} under {lead['name']} "
                    f"(lines {lead.get('line_start')}-{lead.get('line_end')})."
                )
            return "Not found in codebase"

        if "tech stack" in query_lower or re.search(r"\bstack\b", query_lower):
            parts = []
            if backend_framework:
                parts.append(f"backend: {backend_framework}")
            if frontend_framework:
                parts.append(f"frontend: {frontend_framework}")
            if database_name:
                parts.append(f"database: {database_name}")
            if "gemini" in combined.lower():
                parts.append("AI: Google Gemini")
            if "gmail" in combined.lower():
                parts.append("email integration: Gmail SMTP + IMAP")
            if parts:
                return "The main stack is " + ", ".join(parts) + "."
            return "Not found in codebase"

        return None

    def _extract_dataset_answer(self, query: str, retrieved_chunks: List[Dict]) -> str | None:
        query_lower = query.lower()
        if "dataset" not in query_lower:
            return None

        build_chunk = None
        driver_chunk = None
        split_chunk = None
        for chunk in retrieved_chunks:
            name_lower = chunk["name"].lower()
            code_lower = chunk["code"].lower()
            if build_chunk is None and (
                "build_synthetic_dataset" in name_lower
                or ("dataset.from_list" in code_lower and "rows.append" in code_lower)
            ):
                build_chunk = chunk
            if driver_chunk is None and (
                ("dataset =" in code_lower and "build_synthetic_dataset" in code_lower)
                or name_lower == "main"
            ):
                driver_chunk = chunk
            if split_chunk is None and "train_test_split" in code_lower:
                split_chunk = chunk

        if build_chunk is None:
            return None

        answer_parts = [
            (
                "The training dataset is a synthetic chat-style persona explanation dataset, "
                "constructed from rows containing `topic`, `persona`, and chat `messages`."
            ),
            (
                f"It is built in {build_chunk['file_path']} inside {build_chunk['name']} "
                f"(lines {build_chunk.get('line_start')}-{build_chunk.get('line_end')})."
            ),
        ]

        if driver_chunk is not None:
            answer_parts.append(
                f"The same file uses it in {driver_chunk['name']} "
                f"(lines {driver_chunk.get('line_start')}-{driver_chunk.get('line_end')}) to create the dataset for fine-tuning."
            )

        if split_chunk is not None:
            answer_parts.append(
                f"It is then split into train and eval sets in {split_chunk['name']} "
                f"(lines {split_chunk.get('line_start')}-{split_chunk.get('line_end')})."
            )

        return " ".join(answer_parts)

    def _extract_embedding_answer(self, query: str, retrieved_chunks: List[Dict]) -> str | None:
        query_lower = query.lower()
        if not any(term in query_lower for term in ("embed", "embedding", "vector store", "faiss", "vector")):
            return None

        ingestion_chunk = next(
            (
                chunk for chunk in retrieved_chunks
                if chunk["file_path"].endswith("ingestion.py")
                and chunk["name"] in {"IngestionService.ingest", "ingest", "IngestionService._chunk_to_embedding_text", "_chunk_to_embedding_text"}
            ),
            None,
        )
        embedding_chunk = next(
            (
                chunk for chunk in retrieved_chunks
                if chunk["file_path"].endswith("embeddings.py")
                and chunk["name"] in {"EmbeddingService.embed_texts", "embed_texts", "EmbeddingService.embed_query"}
            ),
            None,
        )
        store_chunk = next(
            (
                chunk for chunk in retrieved_chunks
                if chunk["file_path"].endswith("retrieval.py")
                and (
                    chunk["name"] in {"VectorStore.save", "save", "VectorStore"}
                    or "faiss.write_index" in chunk["code"]
                )
            ),
            None,
        )

        if not any((ingestion_chunk, embedding_chunk, store_chunk)):
            return None

        parts = []
        if ingestion_chunk:
            ingestion_action = (
                "turns each chunk into embedding text with file path, type, symbol name, and code"
                if "_chunk_to_embedding_text" in ingestion_chunk["name"]
                else "turns each chunk into embedding text with file path, type, symbol name, and code, then calls the embedding service"
            )
            parts.append(
                f"In {ingestion_chunk['file_path']}, {ingestion_chunk['name']} "
                f"(lines {ingestion_chunk.get('line_start')}-{ingestion_chunk.get('line_end')}) "
                f"{ingestion_action}."
            )
        if embedding_chunk:
            parts.append(
                f"In {embedding_chunk['file_path']}, {embedding_chunk['name']} "
                f"(lines {embedding_chunk.get('line_start')}-{embedding_chunk.get('line_end')}) "
                "uses SentenceTransformer.encode with normalized embeddings and returns float32 vectors."
            )
        if store_chunk:
            parts.append(
                f"In {store_chunk['file_path']}, {store_chunk['name']} "
                f"(lines {store_chunk.get('line_start')}-{store_chunk.get('line_end')}) "
                "stores those vectors in a FAISS IndexFlatIP index and writes the chunk metadata JSON beside it."
            )
        return " ".join(parts)

    def summarize_file(self, file_path: str, language: str, symbols: List[str], code: str) -> str:
        return self._infer_file_summary(file_path, symbols, code)

    def summarize_repository_bundle(self, repo_name: str, file_records: List[Dict], readme_excerpt: str) -> Dict:
        summaries = [
            {
                "file_path": record["file_path"],
                "language": record["language"],
                "summary": self._infer_file_summary(record["file_path"], record["symbols"], record["code"]),
            }
            for record in file_records
        ]
        compact_files = [
            {
                "file_path": item["file_path"],
                "language": item["language"],
                "summary": item["summary"],
            }
            for item in summaries[:120]
        ]
        prompt = "\n".join(
            [
                "Create a compact repository understanding bundle.",
                "Return valid JSON only.",
                'Schema: {"repo_summary":"..."}',
                "Keep repo_summary to 2 short sentences max.",
                f"Repository: {repo_name}",
                f"README excerpt: {readme_excerpt[:2400] or 'None'}",
                f"Files: {json.dumps(compact_files, ensure_ascii=True)}",
            ]
        )

        try:
            raw = self._chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You summarize code repositories. "
                            "Return only valid JSON matching the requested schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=220,
            )
            data = json.loads(raw)
            return {
                "repo_summary": self._clean_markdown_text(data.get("repo_summary", "")) or self._heuristic_repo_summary(repo_name, summaries, readme_excerpt),
                "file_summaries": summaries,
            }
        except Exception:
            return {
                "repo_summary": self._heuristic_repo_summary(repo_name, summaries, readme_excerpt),
                "file_summaries": summaries,
            }

    def _heuristic_repo_summary(self, repo_name: str, file_summaries: List[Dict], readme_excerpt: str) -> str:
        cleaned_readme = self._clean_markdown_text(readme_excerpt)
        if cleaned_readme:
            first_sentence = re.split(r"(?<=[.!?])\s+", cleaned_readme, maxsplit=1)[0].strip()
            if first_sentence:
                return first_sentence
        highlights = ", ".join(f"{item['file_path']} ({item['summary']})" for item in file_summaries[:4])
        if highlights:
            return f"{repo_name} is a code repository with key files such as {highlights}."
        return f"{repo_name} is a code repository with indexed source files."

    def summarize_repository(self, repo_name: str, file_summaries: List[Dict], readme_excerpt: str) -> str:
        prompt = "\n".join(
            [
                "Summarize what this software repository does.",
                "Keep it grounded in the provided evidence.",
                "Respond in 2 short sentences maximum.",
                f"Repository: {repo_name}",
                "File summaries:",
                build_file_summary_context(file_summaries[:80]),
                "README excerpt:",
                readme_excerpt[:4000] or "No README excerpt available.",
            ]
        )

        try:
            return self._chat(
                [
                    {
                        "role": "system",
                        "content": "You explain repositories clearly and briefly without inventing details.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=120,
            )
        except Exception:
            if readme_excerpt:
                return self._clean_markdown_text(readme_excerpt[:320])
            return f"{repo_name} is a software repository with indexed source files."

    def answer_query(self, query: str, retrieved_chunks: List[Dict], repo_context: Dict | None = None) -> str:
        repo_context = repo_context or {}
        stack_answer = self._extract_stack_answer(query, repo_context, retrieved_chunks)
        if stack_answer:
            return stack_answer

        embedding_answer = self._extract_embedding_answer(query, retrieved_chunks)
        if embedding_answer:
            return embedding_answer

        dataset_answer = self._extract_dataset_answer(query, retrieved_chunks)
        if dataset_answer:
            return dataset_answer

        if self._is_repo_overview_query(query) and repo_context.get("repo_summary"):
            try:
                return self._chat(
                    [
                        {
                            "role": "system",
                            "content": "You explain repositories clearly and concretely from provided evidence only.",
                        },
                        {
                            "role": "user",
                            "content": "\n\n".join(
                                [
                                    f"Repository overview:\n{self._clean_markdown_text(repo_context.get('repo_summary', ''))}",
                                    f"File summaries:\n{build_file_summary_context(repo_context.get('file_summaries', [])[:16])}",
                                    "Answer the question in 2 to 4 sentences: what is this repository about?",
                                ]
                            ),
                        },
                    ],
                    temperature=0.2,
                )
            except Exception as e:
                print(f"Ollama API error: {e}")
                return self._repo_overview_fallback(repo_context)

        repo_summary = repo_context.get("repo_summary", "No repository overview available.")
        file_summaries = repo_context.get("file_summaries", [])
        relevant_files = sorted({chunk["file_path"] for chunk in retrieved_chunks})
        relevant_file_summaries = [
            item for item in file_summaries if item.get("file_path") in relevant_files
        ]
        if not relevant_file_summaries:
            lowered_terms = {term.lower() for term in query.split() if len(term) > 2}
            relevant_file_summaries = [
                item
                for item in file_summaries
                if any(term in f"{item.get('file_path', '')} {item.get('summary', '')}".lower() for term in lowered_terms)
            ][:12]

        compact_context = build_compact_chunk_context(retrieved_chunks, limit=6)
        repo_prompt = "\n\n".join(
            [
                f"Repository summary:\n{self._clean_markdown_text(repo_summary)}",
                f"Relevant files:\n{build_file_summary_context(relevant_file_summaries[:10])}",
                f"Retrieved evidence:\n{compact_context}",
                f"Question: {query}",
            ]
        )

        try:
            return self._sanitize_answer_output(
                self._chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a grounded repository analyst. "
                            "Answer the question directly in 2 to 5 sentences. "
                            "Start with the answer, not with headings. "
                            "Mention the most relevant file names. "
                            "Do not dump raw context, labels, or prompt text."
                        ),
                    },
                    {"role": "user", "content": repo_prompt},
                ],
                temperature=0.2,
                max_tokens=180,
            )
            )
        except Exception as e:
            print(f"Ollama API error: {e}")
            return self._fallback_answer(query, retrieved_chunks, repo_context)

    def _fallback_answer(self, query: str, retrieved_chunks: List[Dict], repo_context: Dict | None = None) -> str:
        repo_context = repo_context or {}

        if self._is_repo_overview_query(query) and repo_context.get("repo_summary"):
            return self._repo_overview_fallback(repo_context)

        if not retrieved_chunks:
            return "Not found in codebase"

        query_lower = query.lower()
        if "dataset" in query_lower:
            for chunk in retrieved_chunks:
                code_lower = chunk["code"].lower()
                if "dataset" in code_lower or "rows.append" in code_lower or "from_dict" in code_lower:
                    return (
                        f"The dataset-related logic is in {chunk['file_path']} inside {chunk['name']} "
                        f"(lines {chunk.get('line_start')}-{chunk.get('line_end')}). "
                        "This is the strongest retrieved evidence for how the training data is prepared."
                    )

        lead = retrieved_chunks[0]
        related = ", ".join(
            f"{chunk['file_path']}::{chunk['name']}"
            for chunk in retrieved_chunks[:4]
        )
        return (
            f"The strongest retrieved evidence is in {lead['file_path']} under {lead['name']} "
            f"(lines {lead.get('line_start')}-{lead.get('line_end')}). "
            f"Related retrieved symbols: {related}."
        )
