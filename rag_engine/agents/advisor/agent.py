"""Agent tạo câu trả lời tư vấn dựa trên ngữ cảnh đã truy xuất."""

from rag_engine.agents.state import AgentState
from rag_engine.core.llm import generate_response
from rag_engine.core.prompt_loader import load_prompt


SMALLTALK_SYSTEM = (
    "Bạn là trợ lý tư vấn sản phẩm công nghệ, thân thiện và ngắn gọn. "
    "Người dùng vừa gửi một câu chào hoặc câu hỏi xã giao. "
    "Hãy đáp lại tự nhiên bằng 1-2 câu, sau đó mời họ hỏi về sản phẩm "
    "(laptop, điện thoại, linh kiện...) để bạn tư vấn.\n\n"
    "Câu của người dùng: {query}\n\nTrả lời:"
)


def advisor_agent(state: AgentState) -> AgentState:
    """Sinh câu trả lời cuối cho người dùng từ prompt, context và query."""
    prompt = load_prompt().format(
        context=state.get("context", ""),
        query=state["query"],
    )
    answer = generate_response(prompt, temperature=float(state.get("temperature", 0.1)))
    return {**state, "answer": answer}


def smalltalk_agent(state: AgentState) -> AgentState:
    """Trả lời câu xã giao mà không cần RAG."""
    prompt = SMALLTALK_SYSTEM.format(query=state["query"])
    answer = generate_response(prompt, temperature=float(state.get("temperature", 0.5)))
    return {**state, "answer": answer, "sources": []}
