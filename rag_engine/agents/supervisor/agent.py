from rag_engine.agents.state import AgentState


def supervisor_agent(state: AgentState) -> AgentState:
    query = state.get("query", "").strip()
    if not query:
        return {
            **state,
            "route": "invalid",
            "error": "Query is empty.",
            "answer": "Vui lòng nhập câu hỏi về sản phẩm cần tư vấn.",
        }

    return {**state, "query": query, "route": "product_advice"}

