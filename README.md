![Demo](images/image.png)

# Tech Chatbot RAG Multi-Agent

Ứng dụng Django + LangGraph cho chatbot tư vấn sản phẩm công nghệ dựa trên dữ liệu CSV nội bộ. Hệ thống dùng RAG với FAISS, embedding HuggingFace và hỗ trợ cả Gemini lẫn Ollama để trả lời về điện thoại, laptop, iPad hoặc thông số sản phẩm.

## Nhiệm vụ chính

- Nạp dữ liệu sản phẩm từ `data/*.csv`.
- Chia nhỏ dữ liệu thành chunks, embedding và lưu vào FAISS index.
- Dùng LangGraph để điều phối nhiều agent: phân luồng, truy xuất dữ liệu, sinh câu trả lời và kiểm tra guardrail.
- Không bịa thông tin khi không tìm thấy context nội bộ phù hợp.
- Cho Django gọi RAG engine qua service thay vì trộn logic AI vào views.

## Cấu trúc production

```text
tech_chatbot_rag_multiagent_app/
├── manage.py
├── requirements.txt
├── tech_chatbot_rag_multiagent_app/   # Django project settings/urls/asgi/wsgi
├── chat/                              # Django app/API layer
│   ├── services.py                    # Adapter gọi rag_engine
│   ├── views.py                       # POST /chat/message/
│   └── management/commands/
│       └── build_rag_index.py         # Build FAISS index offline
├── rag_engine/                        # Framework-neutral AI package
│   ├── agents/                        # LangGraph multi-agent workflow
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── supervisor/README.md
│   │   ├── retrieval/README.md
│   │   ├── advisor/README.md
│   │   └── guardrails/README.md
│   ├── core/                          # Config, LLM, embedding, prompt loading
│   ├── ingestion/                     # Offline data/index pipeline
│   └── rag/                           # Loader, chunking, vector store, retriever
├── prompts/
│   └── rag_prompt.txt
├── data/
├── storage/
│   └── faiss_index/                   # Generated, ignored by Git
└── crawler/
```

## Agent workflow

```text
supervisor -> retrieval -> advisor -> final_guardrails -> END
                 └-------> no_context_guardrails -> END
```

- `supervisor`: kiểm tra query và chọn luồng tư vấn sản phẩm.
- `retrieval`: tìm context liên quan trong FAISS.
- `advisor`: dùng prompt RAG và LLM đã cấu hình (`Gemini` hoặc `Ollama`) để sinh câu trả lời.
- `guardrails`: chặn câu trả lời khi thiếu context hoặc output rỗng.

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Tạo `.env`:

```env
LLM_PROVIDER=auto
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qcwind/qwen2.5-7B-instruct-Q4_K_M
RAG_TOP_K=10
RAG_TEMPERATURE=0.1
```

## Chế độ LLM

- `LLM_PROVIDER=auto`: ưu tiên Gemini nếu có `GOOGLE_API_KEY`, và tự động chuyển sang Ollama nếu Gemini lỗi hoặc không dùng được.
- `LLM_PROVIDER=gemini`: ưu tiên Gemini trước, nhưng vẫn có thể rơi sang Ollama khi Gemini không hoạt động.
- `LLM_PROVIDER=ollama`: ưu tiên Ollama trước, nhưng vẫn có thể rơi sang Gemini khi Ollama không hoạt động và `GOOGLE_API_KEY` đã được cấu hình.

## Build index và chạy Django

```bash
python manage.py build_rag_index
python manage.py runserver
```

Gửi câu hỏi:

```bash
curl -X POST http://127.0.0.1:8000/chat/message/ ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Tư vấn điện thoại pin tốt trong tầm giá hợp lý\"}"
```

Response:

```json
{
  "answer": "...",
  "sources": ["..."],
  "error": null
}
```
