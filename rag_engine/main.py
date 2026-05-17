"""Entry point chạy chatbot RAG ở chế độ dòng lệnh."""

import os
import traceback

from rag_engine.core.config import settings
from rag_engine.core.prompt_loader import load_prompt
from rag_engine.rag.chunking import split_docs
from rag_engine.rag.loader import load_data
from rag_engine.rag.pipeline import ask_rag
from rag_engine.rag.vector_store import (
    count_vectors,
    create_vector_db,
    load_vector_db,
)


def prepare_chunks():
    """Load dữ liệu nguồn và chia thành các chunk nhỏ để đưa vào vector DB."""
    docs = load_data()
    return split_docs(docs)


def get_or_create_db(chunks):
    """Load vector DB hiện có hoặc tạo mới khi chưa tồn tại (đa backend)."""
    try:
        print("Loading existing vector DB...")
        return load_vector_db(settings.faiss_index_dir)
    except ValueError:
        print("No index found. Creating new vector DB...")
        return create_vector_db(chunks, settings.faiss_index_dir)


def update_db_if_needed(db, chunks):
    """Bổ sung chunk mới khi dữ liệu nguồn tăng lên (chỉ áp dụng cho FAISS)."""
    if settings.vector_backend.lower() == "qdrant":
        print("Qdrant: upsert thực hiện trực tiếp khi build_index, bỏ qua bước update.")
        return

    try:
        current_count = count_vectors(db)
        new_count = len(chunks)

        if new_count > current_count:
            print(f"Adding {new_count - current_count} new chunks...")
            db.add_documents(chunks[current_count:])
            db.save_local(str(settings.faiss_index_dir))
            print("DB updated.")
        else:
            print("DB is up to date.")
    except Exception as exc:
        print("Update DB error:", exc)
        traceback.print_exc()


def run_chatbot(db):
    """Chạy vòng lặp chat CLI và trả lời từng câu hỏi bằng RAG pipeline."""
    print("\nChatbot ready. Type 'exit' to quit.\n")

    while True:
        query = input("Bạn: ")

        if query.lower() in ["exit", "quit"]:
            print("Bye.")
            break

        try:
            answer = ask_rag(db, query)
            print("Bot:", answer, "\n")
        except Exception as exc:
            print("Error:", exc)


def main():
    """Khởi động chatbot: load prompt, chuẩn bị index và mở vòng lặp chat."""
    print("Starting RAG Chatbot with the configured LLM provider...\n")
    load_prompt()
    chunks = prepare_chunks()
    db = get_or_create_db(chunks)
    update_db_if_needed(db, chunks)
    run_chatbot(db)


if __name__ == "__main__":
    main()
