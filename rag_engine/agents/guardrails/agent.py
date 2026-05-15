"""Các guardrail đảm bảo chatbot luôn trả về câu trả lời hợp lệ."""

from rag_engine.agents.state import AgentState


NO_CONTEXT_ANSWER = (
    "Mình chưa tìm thấy dữ liệu nội bộ đủ liên quan để trả lời chính xác. "
    "Bạn có thể hỏi rõ hơn về tên sản phẩm, nhu cầu, ngân sách hoặc thông số cần so sánh."
)


def no_context_guardrail_agent(state: AgentState) -> AgentState:
    """Trả lời mặc định khi hệ thống không tìm thấy context phù hợp."""
    return {**state, "answer": NO_CONTEXT_ANSWER, "error": "No retrieved context."}


def final_guardrail_agent(state: AgentState) -> AgentState:
    """Kiểm tra câu trả lời cuối và fallback nếu câu trả lời bị rỗng."""
    answer = (state.get("answer") or "").strip()
    if not answer:
        return no_context_guardrail_agent(state)
    return {**state, "answer": answer}
