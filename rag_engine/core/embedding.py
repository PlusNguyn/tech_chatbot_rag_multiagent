"""Khởi tạo embedding model dùng để mã hóa tài liệu và truy vấn."""

from functools import lru_cache

from rag_engine.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    """Trả về HuggingFace embeddings đã cache để tái sử dụng trong RAG."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
