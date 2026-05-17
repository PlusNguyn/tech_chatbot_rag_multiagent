"""Khoi tao embedding model dung de ma hoa tai lieu va truy van."""

from functools import lru_cache
import logging

from rag_engine.core.config import settings
from rag_engine.core.huggingface import resolve_model_source

logger = logging.getLogger(__name__)


def resolve_torch_device(requested_device: str) -> str:
    """Chon thiet bi torch tu cau hinh, uu tien GPU neu kha dung."""
    normalized_device = (requested_device or "auto").strip().lower()

    try:
        import torch
    except ImportError:
        if normalized_device not in {"auto", "cpu"}:
            logger.warning(
                "Torch khong kha dung cho device '%s', fallback ve CPU.",
                normalized_device,
            )
        return "cpu"

    if normalized_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "EMBEDDING_DEVICE=%s nhung CUDA khong kha dung, fallback ve CPU.",
            normalized_device,
        )
        return "cpu"

    return normalized_device


@lru_cache(maxsize=1)
def get_embeddings():
    """Tra ve HuggingFace embeddings da cache de tai su dung trong RAG."""
    from langchain_huggingface import HuggingFaceEmbeddings

    device = resolve_torch_device(settings.embedding_device)
    model_source, local_files_only = resolve_model_source(settings.embedding_model)
    logger.info(
        "Loading embedding model '%s' from '%s' on device '%s'.",
        settings.embedding_model,
        model_source,
        device,
    )

    return HuggingFaceEmbeddings(
        model_name=model_source,
        model_kwargs={
            "device": device,
            "local_files_only": local_files_only,
        },
        encode_kwargs={"normalize_embeddings": True},
    )
