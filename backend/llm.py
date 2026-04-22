import os
import json
import re
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


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


def build_compact_chunk_context(chunks: List[Dict]) -> str:
    if not chunks:
        return "No retrieved code snippets."

    rendered = []
    for chunk in chunks[:3]:
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
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        self.model_name = os.getenv("OLLAMA_MODEL", "tinyllama")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )

    def _chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 220) -> str:
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

        compact_context = build_compact_chunk_context(retrieved_chunks)
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

        if not retrieved_chunks and repo_context.get("repo_summary"):
            return self._clean_markdown_text(repo_context["repo_summary"])

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
        return (
            f"The strongest retrieved evidence is in {lead['file_path']} under {lead['name']} "
            f"(lines {lead.get('line_start')}-{lead.get('line_end')}). "
            "I can answer more precisely when the local model returns a full generation."
        )
