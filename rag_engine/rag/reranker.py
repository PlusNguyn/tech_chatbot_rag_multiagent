"""Tien ich rerank tai lieu truoc khi dua context vao LLM."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document

from rag_engine.core.config import settings
from rag_engine.core.embedding import resolve_torch_device
from rag_engine.core.huggingface import resolve_model_source

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional dependency guard
    CrossEncoder = None


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_reranker():
    """Lazy-load CrossEncoder de tranh khoi dong cham neu khong dung rerank."""
    if not settings.rerank_enabled or CrossEncoder is None:
        return None

    device = resolve_torch_device(settings.embedding_device)
    model_source, local_files_only = resolve_model_source(settings.rerank_model)
    logger.info(
        "Loading reranker '%s' from '%s' on device '%s'.",
        settings.rerank_model,
        model_source,
        device,
    )
    return CrossEncoder(
        model_source,
        device=device,
        local_files_only=local_files_only,
    )


def get_candidate_k(top_k: int) -> int:
    """So luong ung vien can lay tu vector store truoc khi rerank."""
    top_k = max(1, int(top_k))
    if not settings.rerank_enabled:
        return top_k
    return max(top_k, int(settings.rerank_candidate_k))


def rerank_documents(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Sap xep lai docs bang cross-encoder va cat xuong top_k cuoi cung.

    Neu reranker khong san sang, ham se fallback ve thu tu retrieval ban dau.
    """
    if not docs:
        return []

    final_top_k = max(1, min(int(top_k), len(docs)))
    if not settings.rerank_enabled or len(docs) <= 1:
        return docs[:final_top_k]

    model = None
    try:
        model = _load_reranker()
    except Exception as exc:  # pragma: no cover - depends on local model availability
        logger.warning("Khong the khoi tao reranker '%s': %s", settings.rerank_model, exc)

    if model is None:
        return docs[:final_top_k]

    try:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
    except Exception as exc:  # pragma: no cover - depends on local model availability
        logger.warning("Rerank that bai, giu thu tu retrieval ban dau: %s", exc)
        return docs[:final_top_k]

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    reranked: list[Document] = []
    for doc, score in ranked_docs[:final_top_k]:
        metadata = dict(doc.metadata or {})
        metadata["rerank_score"] = float(score)
        reranked.append(Document(page_content=doc.page_content, metadata=metadata))

    return reranked
