"""Hàm truy xuất tài liệu liên quan từ vector store."""

def retrieve(db, query, k=10):
    """Tìm k tài liệu gần nhất với query bằng similarity search."""
    return db.similarity_search(query, k=k)
