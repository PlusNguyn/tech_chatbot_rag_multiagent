# Workflow Multi-Agent RAG Chatbot

Tài liệu mô tả kiến trúc và luồng xử lý của chatbot tư vấn sản phẩm, được xây dựng trên **LangGraph** với 5 agent.

## 1. Tổng quan

Project dùng LangGraph để tạo một workflow dạng đồ thị (graph) gồm nhiều node, mỗi node là một **agent** — hàm Python nhận `AgentState` và trả về `AgentState` mới. Các agent chia sẻ dữ liệu qua một dict trạng thái chung.

File điều phối chính: [rag_engine/agents/graph.py](rag_engine/agents/graph.py)

## 2. AgentState — bộ nhớ chung

Định nghĩa tại [rag_engine/agents/state.py](rag_engine/agents/state.py).

| Trường            | Ý nghĩa                                              |
| ----------------- | ---------------------------------------------------- |
| `query`           | Câu hỏi của người dùng                               |
| `route`           | Hướng đi tiếp theo (`product_advice` / `invalid`)    |
| `retrieved_docs`  | Danh sách `Document` lấy từ vector store             |
| `context`         | Chuỗi ghép từ nội dung các doc                       |
| `sources`         | Danh sách nguồn (unique) của các doc                 |
| `answer`          | Câu trả lời cuối cùng                                |
| `error`           | Thông báo lỗi (nếu có)                               |
| `top_k`           | Số doc cần lấy khi retrieval                         |
| `temperature`     | Độ "sáng tạo" của LLM (mặc định 0.1)                 |

Mỗi agent trả về `{**state, ...}` — tạo state mới thay vì mutate, giúp dễ trace luồng dữ liệu.

## 3. Các Agent

### 3.1. `supervisor_agent`
File: [rag_engine/agents/supervisor/agent.py](rag_engine/agents/supervisor/agent.py)

- **Vai trò:** người gác cổng đầu vào.
- **Logic:**
  - Strip `query`.
  - Nếu rỗng → `route="invalid"`, set sẵn `answer` mặc định → đi tới `final_guardrails`.
  - Nếu hợp lệ → `route="product_advice"` → đi tới `retrieval`.

### 3.2. `retrieval_agent`
File: [rag_engine/agents/retrieval/agent.py](rag_engine/agents/retrieval/agent.py)

- **Vai trò:** truy xuất tài liệu liên quan từ vector store.
- **Pattern:** dùng factory `make_retrieval_agent(db, default_top_k)` để "tiêm" sẵn `db` và `top_k`, vì node của LangGraph chỉ nhận `state`.
- **Logic:**
  - Gọi `retrieve(db, query, k=top_k)` → list `Document`.
  - Ghép `page_content` thành chuỗi `context`.
  - Trích `sources` unique từ `metadata.source`.

### 3.3. `advisor_agent`
File: [rag_engine/agents/advisor/agent.py](rag_engine/agents/advisor/agent.py)

- **Vai trò:** sinh câu trả lời tư vấn.
- **Logic:**
  - Load prompt template qua `load_prompt()`.
  - Format prompt với `context` + `query`.
  - Gọi `generate_response(prompt, temperature)` để lấy `answer` từ LLM.

### 3.4. `no_context_guardrail_agent`
File: [rag_engine/agents/guardrails/agent.py](rag_engine/agents/guardrails/agent.py)

- **Vai trò:** fallback khi không tìm được tài liệu liên quan.
- Trả câu trả lời cố định, tránh để LLM "bịa" khi thiếu dữ liệu.

### 3.5. `final_guardrail_agent`
File: [rag_engine/agents/guardrails/agent.py](rag_engine/agents/guardrails/agent.py)

- **Vai trò:** kiểm tra cuối.
- Nếu `answer` rỗng → fallback sang `no_context_guardrail_agent`.
- Đảm bảo user luôn nhận được phản hồi hợp lệ.

## 4. Sơ đồ luồng

```
                       supervisor
                      /          \
        route=invalid             route=product_advice
              |                          |
              |                       retrieval
              |                      /         \
              |          không có docs        có docs
              |                |                |
              |       no_context_guardrails  advisor
              |                |                |
              |               END         final_guardrails
              |                                 |
              +------------------------------> END
```

### Conditional edges

Có 2 điểm rẽ nhánh trong graph:

1. **Sau `supervisor`** — hàm `_route_after_supervisor`:
   - `state["route"] == "product_advice"` → `retrieval`
   - ngược lại → `final_guardrails`

2. **Sau `retrieval`** — hàm `_route_after_retrieval`:
   - có `retrieved_docs` → `advisor`
   - không có → `no_context_guardrails`

### Static edges
- `advisor` → `final_guardrails`
- `no_context_guardrails` → `END`
- `final_guardrails` → `END`

## 5. Vì sao thiết kế như vậy?

1. **Tách trách nhiệm:** mỗi agent một việc → dễ test, dễ thay thế.
2. **Guardrail 2 tầng** (no_context + final) → chatbot không bao giờ im lặng hoặc bịa khi thiếu dữ liệu.
3. **Factory cho retrieval** → cùng một graph có thể dùng với nhiều DB khác nhau khi test.
4. **State immutable-ish** (`{**state, ...}`) → dễ debug, dễ trace.
5. **Temperature thấp (0.1)** ở advisor → ưu tiên trung thực với context, hạn chế hallucination.

## 6. Điểm khởi tạo

```python
from rag_engine.agents.graph import build_chat_graph

graph = build_chat_graph(db, top_k=5)
result = graph.invoke({"query": "Laptop nào phù hợp cho sinh viên?"})
print(result["answer"])
print(result["sources"])
```
