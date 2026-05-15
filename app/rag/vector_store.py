import os
from langchain_community.vectorstores import FAISS
from app.core.embedding import embeddings


def create_vector_db(chunks):
    batch_size = 32 

    print(f"Bắt đầu tạo Vector DB cho {len(chunks)} đoạn...")

    # 1. Tạo DB với batch đầu
    db = FAISS.from_documents(chunks[:batch_size], embeddings)

    # 2. Add batch còn lại
    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        print(f"Đang xử lý batch {i} → {i + len(batch)}...")
        db.add_documents(batch)

    # 3. Lưu DB
    db.save_local("faiss_index")
    print("Đã lưu Vector DB")

    return db


def load_vector_db():
    if not os.path.exists("faiss_index"):
        raise ValueError("Chưa có FAISS index")

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )