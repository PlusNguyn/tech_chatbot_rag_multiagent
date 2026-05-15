"""Supervisor agent: phân loại intent và quyết định route cho graph."""

import re

from rag_engine.agents.state import AgentState


SMALLTALK_PATTERNS = [
    r"\bxin ch[aà]o\b",
    r"\bch[aà]o\b",
    r"\bhello\b",
    r"\bhi\b",
    r"\bhey\b",
    r"\bc[ảa]m [ơo]n\b",
    r"\bthanks?\b",
    r"\bthank you\b",
    r"\bt[aạ]m bi[eệ]t\b",
    r"\bbye\b",
    r"\bgoodbye\b",
    r"\bb[aạ]n l[aà] ai\b",
    r"\bb[aạ]n t[eê]n g[ìi]\b",
    r"\bgi[ơớ]i thi[eệ]u v[eề] b[aạ]n\b",
    r"\bwho are you\b",
    r"\bwhat can you do\b",
    r"\bb[aạ]n c[oó] th[eể] l[aà]m g[ìi]\b",
    r"\bkh[oỏ]e kh[oô]ng\b",
    r"\bhow are you\b",
]

PRODUCT_HINTS = [
    "laptop", "máy tính", "pc", "điện thoại", "smartphone", "iphone",
    "samsung", "tai nghe", "chuột", "bàn phím", "màn hình", "monitor",
    "card đồ họa", "gpu", "cpu", "ram", "ssd", "ổ cứng", "linh kiện",
    "giá", "cấu hình", "thông số", "so sánh", "mua", "tư vấn", "gaming",
    "văn phòng", "đồ họa", "thiết kế", "ngân sách",
]


def _looks_like_smalltalk(query: str) -> bool:
    """Trả về True nếu câu hỏi mang tính chào hỏi/social, không phải hỏi sản phẩm."""
    low = query.lower().strip()
    if any(hint in low for hint in PRODUCT_HINTS):
        return False
    if len(low.split()) <= 5 and any(re.search(p, low) for p in SMALLTALK_PATTERNS):
        return True
    return False


def supervisor_agent(state: AgentState) -> AgentState:
    """Chuẩn hóa query và phân loại intent: invalid / smalltalk / product_advice."""
    query = state.get("query", "").strip()
    if not query:
        return {
            **state,
            "route": "invalid",
            "error": "Query is empty.",
            "answer": "Vui lòng nhập câu hỏi về sản phẩm cần tư vấn.",
        }

    if _looks_like_smalltalk(query):
        return {**state, "query": query, "route": "smalltalk"}

    return {**state, "query": query, "route": "product_advice"}
