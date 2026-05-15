"""Agent tạo câu trả lời tư vấn dựa trên ngữ cảnh đã truy xuất."""

from rag_engine.agents.state import AgentState
from rag_engine.core.llm import generate_response
from rag_engine.core.prompt_loader import load_prompt


def advisor_agent(state: AgentState) -> AgentState:
    """Sinh câu trả lời cuối cho người dùng từ prompt, context và query."""
    prompt = load_prompt().format(
        context=state.get("context", ""),
        query=state["query"],
    )
    answer = generate_response(prompt, temperature=float(state.get("temperature", 0.1)))
    return {**state, "answer": answer}
