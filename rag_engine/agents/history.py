"""Tien ich xu ly lich su hoi thoai cho chat theo session."""

from __future__ import annotations

from typing import Any


FOLLOW_UP_MARKERS = (
    "con do",
    "may do",
    "cai do",
    "ban do",
    "loai do",
    "san pham do",
    "cai nay",
    "con nay",
    "may nay",
    "cai nao",
    "con nao",
    "loai nao",
    "so voi",
    "so sanh",
    "re hon",
    "tot hon",
    "manh hon",
    "chi tiet hon",
    "them thong tin",
    "the con",
    "vay con",
    "vay thi",
)


def normalize_history(raw_history: Any, max_messages: int = 8) -> list[dict[str, str]]:
    """Chuan hoa lich su chat tu payload client."""
    if not isinstance(raw_history, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in raw_history[-max_messages:]:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        if role == "bot":
            role = "assistant"
        if role not in {"user", "assistant"}:
            continue

        content = str(item.get("content", "")).strip()
        if not content:
            continue

        normalized.append({"role": role, "content": content})

    return normalized


def format_history_for_prompt(history: list[dict[str, str]], max_messages: int = 6) -> str:
    """Format lich su thanh transcript ngan gon de dua vao prompt."""
    if not history:
        return "Khong co."

    lines: list[str] = []
    for item in history[-max_messages:]:
        speaker = "Nguoi dung" if item["role"] == "user" else "Tro ly"
        lines.append(f"{speaker}: {item['content']}")
    return "\n".join(lines)


def build_retrieval_query(
    query: str,
    history: list[dict[str, str]],
    max_messages: int = 4,
) -> str:
    """Mo rong query neu cau hoi hien tai dang tham chieu lich su chat."""
    normalized_query = query.strip()
    if not normalized_query:
        return normalized_query

    lowered_query = normalized_query.lower()
    needs_history = (
        len(normalized_query.split()) <= 10
        or any(marker in lowered_query for marker in FOLLOW_UP_MARKERS)
    )
    if not history or not needs_history:
        return normalized_query

    recent_lines: list[str] = []
    for item in history[-max_messages:]:
        speaker = "Nguoi dung" if item["role"] == "user" else "Tro ly"
        recent_lines.append(f"{speaker}: {item['content']}")

    recent_context = "\n".join(recent_lines)
    return (
        "Lich su hoi thoai gan day:\n"
        f"{recent_context}\n\n"
        "Cau hoi hien tai cua nguoi dung:\n"
        f"{normalized_query}"
    )
