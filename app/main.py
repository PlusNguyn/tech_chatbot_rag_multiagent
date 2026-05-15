import os
import traceback
from app.rag.loader import load_data
from app.rag.chunking import split_docs
from app.rag.vector_store import create_vector_db, load_vector_db
from app.rag.pipeline import ask_rag
from app.core.prompt_loader import load_prompt


def prepare_chunks():
    docs = load_data()
    return split_docs(docs)


def get_or_create_db(chunks):
    """
    Load DB nếu tồn tại, nếu không thì tạo mới
    """
    if not os.path.exists("faiss_index"):
        print("No index found. Creating new vector DB...")
        return create_vector_db(chunks)
    
    print("Loading existing vector DB...")
    return load_vector_db()


def update_db_if_needed(db, chunks):
    """
    Optional: update nếu có dữ liệu mới
    """
    try:
        current_count = db.index.ntotal
        new_count = len(chunks)

        if new_count > current_count:
            print(f"Adding {new_count - current_count} new chunks...")
            db.add_documents(chunks[current_count:])
            db.save_local("faiss_index")
            print("DB updated!")
        else:
            print("DB is up to date.")
    except Exception as e:
        print("Update DB error:", e)
        traceback.print_exc()


def run_chatbot(db):
    print("\nChatbot ready! (type 'exit' to quit)\n")

    while True:
        query = input("Bạn: ")

        if query.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        try:
            answer = ask_rag(db, query)
            print("Bot:", answer, "\n")
        except Exception as e:
            print("Error:", e)


def main():
    print("Starting RAG Chatbot with Gemini...\n")

    # 1. Check prompt
    try:
        load_prompt()
        print("Prompt OK")
    except Exception as e:
        print("Prompt error:", e)

    # 2. Prepare data
    chunks = prepare_chunks()

    # 3. Load or create DB
    db = get_or_create_db(chunks)

    # 4. Update DB nếu cần
    update_db_if_needed(db, chunks)

    # 5. Run chatbot
    run_chatbot(db)


if __name__ == "__main__":
    main()