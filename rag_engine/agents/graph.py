from langgraph.graph import END, StateGraph

from rag_engine.agents.advisor.agent import advisor_agent
from rag_engine.agents.guardrails.agent import (
    final_guardrail_agent,
    no_context_guardrail_agent,
)
from rag_engine.agents.retrieval.agent import make_retrieval_agent
from rag_engine.agents.state import AgentState
from rag_engine.agents.supervisor.agent import supervisor_agent
from rag_engine.core.config import settings


def _route_after_supervisor(state: AgentState) -> str:
    return "retrieval" if state.get("route") == "product_advice" else "final_guardrails"


def _route_after_retrieval(state: AgentState) -> str:
    return "advisor" if state.get("retrieved_docs") else "no_context_guardrails"


def build_chat_graph(db, top_k: int | None = None):
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("retrieval", make_retrieval_agent(db, top_k or settings.rag_top_k))
    workflow.add_node("advisor", advisor_agent)
    workflow.add_node("no_context_guardrails", no_context_guardrail_agent)
    workflow.add_node("final_guardrails", final_guardrail_agent)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "retrieval": "retrieval",
            "final_guardrails": "final_guardrails",
        },
    )
    workflow.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {
            "advisor": "advisor",
            "no_context_guardrails": "no_context_guardrails",
        },
    )
    workflow.add_edge("advisor", "final_guardrails")
    workflow.add_edge("no_context_guardrails", END)
    workflow.add_edge("final_guardrails", END)

    return workflow.compile()

