def ask_chatbot(query: str) -> dict:
    from rag_engine.rag.pipeline import ask

    return ask(query)
