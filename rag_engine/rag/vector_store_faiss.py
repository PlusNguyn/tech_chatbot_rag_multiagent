"""Backend FAISS: tạo, lưu, load và đếm vector trong FAISS index."""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag_engine.core.config import settings
from rag_engine.core.embedding import get_embeddings


def create_faiss_db(chunks, index_dir=None):
    """Tạo FAISS index từ chunks và lưu xuống ổ đĩa."""
    if not chunks:
        raise ValueError("Cannot create FAISS index from empty chunks.")

    index_path = Path(index_dir or settings.faiss_index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    batch_size = 32

    print(f"Creating FAISS DB for {len(chunks)} chunks...")
    db = FAISS.from_documents(chunks[:batch_size], get_embeddings())

    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Processing batch {i} -> {i + len(batch)}...")
        db.add_documents(batch)

    db.save_local(str(index_path))
    print(f"Saved FAISS DB to {index_path}")
    return db


def load_faiss_db(index_dir=None):
    """Load FAISS index từ ổ đĩa với embedding model hiện tại."""
    index_path = Path(index_dir or settings.faiss_index_dir)
    if not index_path.exists():
        raise ValueError(f"FAISS index not found at {index_path}")

    return FAISS.load_local(
        str(index_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def count_faiss_vectors(db) -> int:
    """Đếm số vector hiện có trong FAISS index."""
    return db.index.ntotal
