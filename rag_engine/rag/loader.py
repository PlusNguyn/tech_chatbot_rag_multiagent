"""Load product TXT data from the local directory into LangChain documents."""

from __future__ import annotations

import re
from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from rag_engine.core.config import settings


SEPARATOR_PATTERN = re.compile(r"\r?\n={20,}(?:\r?\n)+", flags=re.MULTILINE)


def load_data(directory_path: str | Path | None = None) -> list[Document]:
    """Read all TXT product files in the data directory and return documents."""
    data_path = Path(directory_path or settings.data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    txt_files = sorted(
        path
        for path in data_path.rglob("*.txt")
        if path.is_file() and path.name.lower() != "failed_urls.txt"
    )
    if not txt_files:
        raise FileNotFoundError(f"No TXT data files found in: {data_path}")

    docs: list[Document] = []
    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        records = [record.strip() for record in SEPARATOR_PATTERN.split(content) if record.strip()]
        if not records:
            records = [content]

        for index, record in enumerate(records, start=1):
            metadata = {
                "source": str(txt_file),
                "record_index": index,
            }
            product_name = _extract_product_name(record)
            if product_name:
                metadata["product_name"] = product_name

            docs.append(Document(page_content=record, metadata=metadata))

    return docs


def _extract_product_name(record: str) -> str | None:
    """Extract the product name from a TXT block when available."""
    for line in record.splitlines():
        stripped = line.strip()
        if stripped and ":" in stripped:
            return stripped.split(":", 1)[1].strip()
    return None
