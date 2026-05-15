"""Kiểu dữ liệu trạng thái được truyền giữa các agent trong LangGraph."""

from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """Schema trạng thái dùng chung cho toàn bộ multi-agent RAG workflow."""

    query: str
    route: NotRequired[Literal["product_advice", "invalid"]]
    retrieved_docs: NotRequired[list[Document]]
    context: NotRequired[str]
    answer: NotRequired[str]
    error: NotRequired[str]
    sources: NotRequired[list[str]]
    top_k: NotRequired[int]
    temperature: NotRequired[float]
