"""Service layer: bundling SSE streaming và non-streaming cho chat API."""

import json

from rag_engine.agents.advisor.agent import SMALLTALK_SYSTEM
from rag_engine.agents.guardrails.agent import NO_CONTEXT_ANSWER
from rag_engine.agents.retrieval.agent import make_retrieval_agent
from rag_engine.agents.supervisor.agent import supervisor_agent
from rag_engine.core.config import settings
from rag_engine.core.llm import stream_response
from rag_engine.core.prompt_loader import load_prompt
from rag_engine.rag.pipeline import get_default_db


def ask_chatbot(query: str) -> dict:
    """Giữ tương thích ngược cho các caller dùng kết quả 1 lần (non-stream)."""
    from rag_engine.rag.pipeline import ask

    return ask(query)


def _build_prompt_and_sources(query: str) -> tuple[str, list[str], str]:
    """Chạy supervisor + retrieval (nếu cần) để dựng prompt và lấy sources.

    Trả về (prompt, sources, route). Nếu route='no_context' thì prompt rỗng
    và caller cần fallback bằng NO_CONTEXT_ANSWER.
    """
    state = {"query": query, "top_k": settings.rag_top_k}
    state = supervisor_agent(state)
    route = state.get("route", "product_advice")

    if route == "invalid":
        return "", [], "invalid"

    if route == "smalltalk":
        return SMALLTALK_SYSTEM.format(query=query), [], "smalltalk"

    db = get_default_db()
    retrieval = make_retrieval_agent(db, settings.rag_top_k)
    state = retrieval(state)

    if not state.get("retrieved_docs"):
        return "", [], "no_context"

    prompt = load_prompt().format(context=state.get("context", ""), query=query)
    return prompt, state.get("sources", []), "product_advice"


def _sse(event: dict) -> str:
    """Đóng gói payload thành một dòng SSE."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def stream_chat(query: str):
    """Generator yield các sự kiện SSE: sources -> token... -> done."""
    try:
        prompt, sources, route = _build_prompt_and_sources(query)
    except Exception as exc:
        yield _sse({"error": str(exc)})
        yield _sse({"done": True})
        return

    if route == "invalid":
        yield _sse({"token": "Vui lòng nhập câu hỏi về sản phẩm cần tư vấn."})
        yield _sse({"done": True, "sources": []})
        return

    if route == "no_context":
        yield _sse({"token": NO_CONTEXT_ANSWER})
        yield _sse({"done": True, "sources": []})
        return

    yield _sse({"sources": sources, "route": route})

    temperature = 0.5 if route == "smalltalk" else settings.rag_temperature
    try:
        for chunk in stream_response(prompt, temperature=temperature):
            yield _sse({"token": chunk})
    except Exception as exc:
        yield _sse({"error": str(exc)})

    yield _sse({"done": True, "sources": sources})
