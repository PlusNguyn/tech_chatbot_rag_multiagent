"""Supervisor agent điều phối hướng xử lý ban đầu của câu hỏi."""

from rag_engine.agents.state import AgentState


def supervisor_agent(state: AgentState) -> AgentState:
    """Chuẩn hóa query và quyết định route tiếp theo cho graph."""
    query = state.get("query", "").strip()
    if not query:
        return {
            **state,
            "route": "invalid",
            "error": "Query is empty.",
            "answer": "Vui lòng nhập câu hỏi về sản phẩm cần tư vấn.",
        }

    return {**state, "query": query, "route": "product_advice"}
