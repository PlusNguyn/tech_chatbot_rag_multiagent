from app.rag.retriever import retrieve
from app.core.llm import generate_response
from app.core.prompt_loader import load_prompt

def ask_rag(db, query, creative_level=0.1):
    docs = retrieve(db, query)
    context = "\n".join([d.page_content for d in docs])
    prompt = load_prompt().format(context=context, query=query)
    return generate_response(prompt, temperature=creative_level)