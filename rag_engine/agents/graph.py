"""Xây dựng workflow LangGraph cho chatbot RAG nhiều agent."""

from langgraph.graph import END, StateGraph

from rag_engine.agents.advisor.agent import advisor_agent, smalltalk_agent
from rag_engine.agents.guardrails.agent import (
    final_guardrail_agent,
    no_context_guardrail_agent,
)
from rag_engine.agents.retrieval.agent import make_retrieval_agent
from rag_engine.agents.reranker.agent import make_reranker_agent
from rag_engine.agents.state import AgentState
from rag_engine.agents.supervisor.agent import supervisor_agent
from rag_engine.core.config import settings


def _route_after_supervisor(state: AgentState) -> str:
    """Chọn bước tiếp theo sau supervisor dựa trên route trong state."""
    route = state.get("route")
    if route == "product_advice":
        return "retrieval"
    if route == "smalltalk":
        return "smalltalk"
    return "final_guardrails"


def _route_after_retrieval(state: AgentState) -> str:
    """Chuyển sang reranker khi có tài liệu, hoặc guardrail khi thiếu context."""
    return "reranker" if state.get("retrieved_docs") else "no_context_guardrails"


def _route_after_reranker(state: AgentState) -> str:
    """Chỉ sinh câu trả lời khi reranker vẫn giữ được context."""
    return "advisor" if state.get("retrieved_docs") else "no_context_guardrails"


def build_chat_graph(db, top_k: int | None = None):
    """Tạo graph hội thoại gồm supervisor, retrieval, reranker, advisor và guardrails."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("retrieval", make_retrieval_agent(db, top_k or settings.rag_top_k))
    workflow.add_node("reranker", make_reranker_agent(top_k or settings.rag_top_k))
    workflow.add_node("advisor", advisor_agent)
    workflow.add_node("smalltalk", smalltalk_agent)
    workflow.add_node("no_context_guardrails", no_context_guardrail_agent)
    workflow.add_node("final_guardrails", final_guardrail_agent)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "retrieval": "retrieval",
            "smalltalk": "smalltalk",
            "final_guardrails": "final_guardrails",
        },
    )
    workflow.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {
            "reranker": "reranker",
            "no_context_guardrails": "no_context_guardrails",
        },
    )
    workflow.add_conditional_edges(
        "reranker",
        _route_after_reranker,
        {
            "advisor": "advisor",
            "no_context_guardrails": "no_context_guardrails",
        },
    )
    workflow.add_edge("advisor", "final_guardrails")
    workflow.add_edge("smalltalk", "final_guardrails")
    workflow.add_edge("no_context_guardrails", END)
    workflow.add_edge("final_guardrails", END)

    return workflow.compile()
