"""Load dữ liệu CSV từ thư mục local thành LangChain Document."""

from pathlib import Path

from langchain_community.document_loaders import CSVLoader, DirectoryLoader

from rag_engine.core.config import settings


def load_data(directory_path=None):
    """Đọc toàn bộ file CSV trong thư mục dữ liệu và trả về Document list."""
    data_path = Path(directory_path or settings.data_dir).resolve()

    loader = DirectoryLoader(
        path=str(data_path),
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "csv_args": {
                "delimiter": ",",
            },
        },
    )

    return loader.load()
