"""Ingestion pipeline to build the vector index from TXT source data."""

from rag_engine.core.config import settings
from rag_engine.rag.chunking import split_docs
from rag_engine.rag.loader import load_data
from rag_engine.rag.vector_store import create_vector_db


def build_index(data_dir=None, index_dir=None):
    """Load data, split it into chunks, and write it to the configured vector store.

    `index_dir` only applies to the FAISS backend. For Qdrant, the target is
    determined by QDRANT_URL and QDRANT_COLLECTION in the configuration.
    """
    docs = load_data(data_dir or settings.data_dir)
    chunks = split_docs(docs)
    return create_vector_db(chunks, index_dir=index_dir or settings.faiss_index_dir)
