from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse


LOGGER_NAME = "codebase_rag"
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".pytest_cache",
}
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


logger = configure_logging()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_github_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and "github.com" in parsed.netloc


def sanitize_repo_name(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    repo_name = parsed.path.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return repo_name or "repository"


def clone_repository(repo_url: str, target_root: Path) -> Path:
    ensure_directory(target_root)
    repo_dir = target_root / sanitize_repo_name(repo_url)

    def handle_remove_readonly(func, path, exc):
        import stat

        os.chmod(path, stat.S_IWRITE)
        func(path)

    git_dir = repo_dir / ".git"
    if git_dir.exists():
        logger.info("Repository already available at %s, reusing existing clone", repo_dir)
        return repo_dir

    if repo_dir.exists():
        logger.info("Removing incomplete repository directory %s", repo_dir)
        shutil.rmtree(repo_dir, onerror=handle_remove_readonly)

    logger.info("Cloning repository %s into %s", repo_url, repo_dir)
    clone_commands = [
        ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
        ["git", "clone", repo_url, str(repo_dir)],
    ]

    last_error: str | None = None
    for attempt, command in enumerate(clone_commands, start=1):
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            return repo_dir
        except subprocess.CalledProcessError as exc:
            error_msg = (exc.stderr or exc.stdout or str(exc)).strip()
            last_error = error_msg or f"git clone exited with status {exc.returncode}"

            if repo_dir.exists():
                shutil.rmtree(repo_dir, onerror=handle_remove_readonly)

            if exc.returncode == 130:
                logger.error("Git clone was interrupted: %s", last_error)
                raise RuntimeError(
                    "Git clone was interrupted before it finished. Please try again."
                ) from exc

            if attempt < len(clone_commands):
                logger.warning("Clone attempt %s failed, retrying with fallback: %s", attempt, last_error)
                continue

    logger.error("Git clone failed: %s", last_error)
    raise RuntimeError(f"Git clone failed: {last_error}")


def resolve_source_path(source: str, repos_root: Path) -> Path:
    source_path = Path(source).expanduser()
    if is_github_url(source):
        return clone_repository(source, repos_root)

    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()

    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Source path does not exist or is not a directory: {source_path}")

    return source_path


def should_skip_dir(path: Path) -> bool:
    return path.name in IGNORE_DIRS


def iter_code_files(root: Path) -> Iterable[Path]:
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = [d for d in dirnames if not should_skip_dir(current_path / d)]

        for filename in filenames:
            file_path = current_path / filename
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield file_path


def detect_language(file_path: Path) -> Optional[str]:
    return SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())


def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def readme_excerpt(root: Path) -> str:
    for candidate in sorted(root.iterdir()) if root.exists() else []:
        if candidate.is_file() and candidate.stem.lower() == "readme":
            return read_text_file(candidate)[:6000]
    return ""


def write_json(path: Path, payload: object) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
