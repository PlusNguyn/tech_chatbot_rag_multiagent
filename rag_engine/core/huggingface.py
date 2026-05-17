"""Tien ich resolve model Hugging Face tu cache local neu da co."""

from __future__ import annotations

import os
from pathlib import Path

from rag_engine.core.config import settings


def _huggingface_hub_dir() -> Path:
    """Tra ve thu muc cache Hugging Face Hub tren may hien tai."""
    hub_cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_model_dir(model_id: str) -> Path:
    """Map model id thanh duong dan cache cua Hugging Face Hub."""
    return _huggingface_hub_dir() / f"models--{model_id.replace('/', '--')}"


def resolve_cached_model_path(model_id: str) -> str | None:
    """Tim snapshot local cua model neu da duoc cache truoc do."""
    if not model_id:
        return None

    as_path = Path(model_id)
    if as_path.exists():
        return str(as_path)

    model_dir = _cached_model_dir(model_id)
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_main = model_dir / "refs" / "main"
    if ref_main.exists():
        snapshot_name = ref_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / snapshot_name
        if snapshot_dir.exists():
            return str(snapshot_dir)

    snapshot_dirs = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshot_dirs:
        return None

    latest_snapshot = max(snapshot_dirs, key=lambda path: path.stat().st_mtime)
    return str(latest_snapshot)


def resolve_model_source(model_id: str) -> tuple[str, bool]:
    """Tra ve nguon model va co can ep local-only hay khong."""
    cached_path = resolve_cached_model_path(model_id)
    if settings.hf_prefer_local_cache and cached_path:
        return cached_path, True

    return model_id, settings.hf_local_files_only
