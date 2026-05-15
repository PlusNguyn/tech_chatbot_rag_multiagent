from pathlib import Path

from rag_engine.core.config import settings


def load_prompt(file_path: str | Path | None = None) -> str:
    prompt_path = Path(file_path) if file_path else settings.prompt_path
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

