"""Chia tài liệu đầu vào thành các chunk phục vụ truy xuất RAG."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(docs):
    """Tách danh sách Document thành các đoạn nhỏ có overlap cố định."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(docs)
