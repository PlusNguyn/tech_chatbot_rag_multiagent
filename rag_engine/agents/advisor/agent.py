"""Agent tao cau tra loi tu van dua tren ngu canh da truy xuat."""

from rag_engine.agents.state import AgentState
from rag_engine.core.llm import generate_response
from rag_engine.core.prompt_loader import load_prompt


SMALLTALK_SYSTEM = (
    "Ban la tro ly tu van san pham cong nghe, than thien va ngan gon. "
    "Nguoi dung vua gui mot cau chao hoac cau hoi xa giao. "
    "Hay dap lai tu nhien bang 1-2 cau, sau do moi ho hoi ve san pham "
    "(laptop, dien thoai, linh kien...) de ban tu van.\n\n"
    "Cau cua nguoi dung: {query}\n\nTra loi:"
)


def advisor_agent(state: AgentState) -> AgentState:
    """Sinh cau tra loi cuoi cho nguoi dung tu prompt, context va query."""
    prompt = load_prompt().format(
        conversation_history=state.get("history_text", "Khong co."),
        context=state.get("context", ""),
        query=state["query"],
    )
    answer = generate_response(prompt, temperature=float(state.get("temperature", 0.1)))
    return {**state, "answer": answer}


def smalltalk_agent(state: AgentState) -> AgentState:
    """Tra loi cau xa giao ma khong can RAG."""
    prompt = SMALLTALK_SYSTEM.format(query=state["query"])
    answer = generate_response(prompt, temperature=float(state.get("temperature", 0.5)))
    return {**state, "answer": answer, "sources": []}
