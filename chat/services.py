"""Service layer for streaming and non-streaming chat responses."""

import json

from rag_engine.agents.advisor.agent import SMALLTALK_SYSTEM
from rag_engine.agents.guardrails.agent import NO_CONTEXT_ANSWER
from rag_engine.agents.history import format_history_for_prompt, normalize_history
from rag_engine.agents.retrieval.agent import make_retrieval_agent
from rag_engine.agents.reranker.agent import make_reranker_agent
from rag_engine.agents.supervisor.agent import supervisor_agent
from rag_engine.core.config import settings
from rag_engine.core.llm import LLMConfigurationError, stream_response
from rag_engine.core.prompt_loader import load_prompt
from rag_engine.rag.pipeline import get_default_db


def ask_chatbot(query: str) -> dict:
    """Backward-compatible non-streaming entrypoint."""
    from rag_engine.rag.pipeline import ask

    return ask(query)


def _build_prompt_and_sources(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[str], str]:
    """Run supervisor and retrieval to construct the prompt and sources."""
    normalized_history = normalize_history(history)
    history_text = format_history_for_prompt(normalized_history)
    state = {
        "query": query,
        "top_k": settings.rag_top_k,
        "history": normalized_history,
        "history_text": history_text,
    }
    state = supervisor_agent(state)
    route = state.get("route", "product_advice")

    if route == "invalid":
        return "", [], "invalid"

    if route == "smalltalk":
        prompt = SMALLTALK_SYSTEM.format(query=query)
        if normalized_history:
            prompt = f"Lich su hoi thoai gan day:\n{history_text}\n\n{prompt}"
        return prompt, [], "smalltalk"

    db = get_default_db()
    retrieval = make_retrieval_agent(db, settings.rag_top_k)
    reranker = make_reranker_agent(settings.rag_top_k)
    state = retrieval(state)
    state = reranker(state)

    if not state.get("retrieved_docs"):
        return "", [], "no_context"

    prompt = load_prompt().format(
        conversation_history=history_text,
        context=state.get("context", ""),
        query=query,
    )
    return prompt, state.get("sources", []), "product_advice"


def _sse(event: dict) -> str:
    """Wrap a payload as one SSE event line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def stream_chat(query: str, history: list[dict[str, str]] | None = None):
    """Yield SSE events: sources, token chunks, and done."""
    try:
        prompt, sources, route = _build_prompt_and_sources(query, history=history)
    except Exception as exc:
        yield _sse({"error": str(exc)})
        yield _sse({"done": True})
        return

    if route == "invalid":
        yield _sse({"token": "Vui long nhap cau hoi ve san pham can tu van."})
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
    except LLMConfigurationError as exc:
        yield _sse({"error": str(exc)})
    except Exception as exc:
        yield _sse({"error": f"Chatbot tam thoi chua the tra loi. Chi tiet: {exc}"})

    yield _sse({"done": True, "sources": sources})
