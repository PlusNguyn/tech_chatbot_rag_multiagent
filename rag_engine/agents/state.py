"""Kieu du lieu trang thai duoc truyen giua cac agent trong LangGraph."""

from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document


class ChatMessage(TypedDict):
    """Mot tin nhan trong lich su hoi thoai."""

    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict):
    """Schema trang thai dung chung cho toan bo multi-agent RAG workflow."""

    query: str
    route: NotRequired[Literal["product_advice", "smalltalk", "invalid"]]
    retrieved_docs: NotRequired[list[Document]]
    context: NotRequired[str]
    answer: NotRequired[str]
    error: NotRequired[str]
    sources: NotRequired[list[str]]
    top_k: NotRequired[int]
    temperature: NotRequired[float]
    history: NotRequired[list[ChatMessage]]
    history_text: NotRequired[str]
    retrieval_query: NotRequired[str]
