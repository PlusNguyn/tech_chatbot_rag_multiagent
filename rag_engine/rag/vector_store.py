"""Dispatcher chọn backend vector store (FAISS hoặc Qdrant) theo cấu hình."""

from rag_engine.core.config import settings
from rag_engine.rag.vector_store_faiss import (
    count_faiss_vectors,
    create_faiss_db,
    load_faiss_db,
)
from rag_engine.rag.vector_store_qdrant import (
    count_qdrant_vectors,
    create_qdrant_db,
    load_qdrant_db,
)


def _backend() -> str:
    """Trả về backend đang dùng, đã chuẩn hóa về chữ thường."""
    return (settings.vector_backend or "faiss").lower()


def create_vector_db(chunks, index_dir=None):
    """Tạo vector DB mới bằng backend đang cấu hình."""
    if _backend() == "qdrant":
        return create_qdrant_db(chunks)
    return create_faiss_db(chunks, index_dir=index_dir)


def load_vector_db(index_dir=None):
    """Load vector DB sẵn có bằng backend đang cấu hình."""
    if _backend() == "qdrant":
        return load_qdrant_db()
    return load_faiss_db(index_dir=index_dir)


def count_vectors(db) -> int:
    """Đếm vector hiện có (đa backend) để báo cáo hoặc so sánh."""
    if _backend() == "qdrant":
        return count_qdrant_vectors(db)
    return count_faiss_vectors(db)
