"""Pipeline ingestion để build FAISS index từ dữ liệu nguồn."""

from rag_engine.core.config import settings
from rag_engine.rag.chunking import split_docs
from rag_engine.rag.loader import load_data
from rag_engine.rag.vector_store import create_vector_db


def build_index(data_dir=None, index_dir=None):
    """Load dữ liệu, chia chunk và lưu thành FAISS vector index."""
    docs = load_data(data_dir or settings.data_dir)
    chunks = split_docs(docs)
    return create_vector_db(chunks, index_dir=index_dir or settings.faiss_index_dir)
