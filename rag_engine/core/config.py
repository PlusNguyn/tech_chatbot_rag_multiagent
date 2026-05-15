"""Cấu hình tập trung cho đường dẫn, model và tham số RAG."""

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Tập hợp cấu hình đọc từ biến môi trường và giá trị mặc định."""

    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    storage_dir: Path = BASE_DIR / "storage"
    faiss_index_dir: Path = BASE_DIR / "storage" / "faiss_index"
    prompt_path: Path = BASE_DIR / "prompts" / "rag_prompt.txt"
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qcwind/qwen2.5-7B-instruct-Q4_K_M")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "10"))
    rag_temperature: float = float(os.getenv("RAG_TEMPERATURE", "0.1"))

    vector_backend: str = os.getenv("VECTOR_BACKEND", "faiss").lower()
    qdrant_url: str | None = os.getenv("QDRANT_URL")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "tech_products")


settings = Settings()
