"""Backend Qdrant Cloud: tạo collection, ingest chunks và load vector store."""

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_engine.core.config import settings
from rag_engine.core.embedding import get_embeddings


def _get_client() -> QdrantClient:
    """Khởi tạo Qdrant client trỏ tới Qdrant Cloud."""
    if not settings.qdrant_url:
        raise ValueError("QDRANT_URL is required when using Qdrant backend.")
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=False,
    )


def _embedding_dim() -> int:
    """Lấy số chiều embedding bằng cách nhúng thử một chuỗi ngắn."""
    return len(get_embeddings().embed_query("dim probe"))


def _ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Tạo collection nếu chưa tồn tại; giữ nguyên nếu đã có."""
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )


def create_qdrant_db(chunks, collection_name: str | None = None):
    """Tạo (hoặc bổ sung) collection Qdrant từ chunks và trả về vector store."""
    if not chunks:
        raise ValueError("Cannot create Qdrant collection from empty chunks.")

    name = collection_name or settings.qdrant_collection
    client = _get_client()
    _ensure_collection(client, name, _embedding_dim())

    store = Qdrant(
        client=client,
        collection_name=name,
        embeddings=get_embeddings(),
    )

    batch_size = 32
    print(f"Ingesting {len(chunks)} chunks into Qdrant collection '{name}'...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"  batch {i} -> {i + len(batch)}")
        store.add_documents(batch)

    print(f"Done. Collection '{name}' updated on Qdrant Cloud.")
    return store


def load_qdrant_db(collection_name: str | None = None):
    """Mở vector store Qdrant cho một collection đã tồn tại."""
    name = collection_name or settings.qdrant_collection
    client = _get_client()

    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        raise ValueError(f"Qdrant collection '{name}' not found.")

    return Qdrant(
        client=client,
        collection_name=name,
        embeddings=get_embeddings(),
    )


def count_qdrant_vectors(db) -> int:
    """Đếm số điểm (vector) hiện có trong collection."""
    info = db.client.get_collection(db.collection_name)
    return info.points_count or 0
