# Tech Chatbot RAG Multi-Agent

![alt text](image.png)

Một hệ thống Chatbot thông minh ứng dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** để tư vấn các sản phẩm công nghệ (Laptop, Điện thoại, iPad) dựa trên dữ liệu nội bộ. Dự án sử dụng mô hình ngôn ngữ lớn **Gemini 2.5** và công cụ tìm kiếm vector **FAISS**.

## Tính năng chính
* **Tư vấn dựa trên dữ liệu thực tế:** Hệ thống đọc dữ liệu từ các file CSV (`tgdd_laptops_FULL.csv`, `soc_ratings.csv`) để trả lời thông số kỹ thuật và giá bán chính xác.
* **Cơ chế RAG tối ưu:** * Sử dụng **RecursiveCharacterTextSplitter** để cắt nhỏ dữ liệu (Chunking) thông minh.
    * Tích hợp **Google Generative AI Embeddings** (`text-embedding-004`).
* **Quản lý Vector Database:** Lưu trữ và nạp chỉ mục cục bộ với **FAISS**, giúp tăng tốc độ truy vấn và tiết kiệm chi phí API.
* **Xử lý giới hạn API (Rate Limit):** Cơ chế **Batching** & **Retry** tự động để vượt qua giới hạn `429 Resource Exhausted` của Google Free Tier.
* **Prompt Engineering:** Tách biệt logic Prompt ra file `.txt` giúp dễ dàng tinh chỉnh "tính cách" chatbot.

## Kiến trúc hệ thống (Workflow)
1. **Data Ingestion:** Load dữ liệu CSV từ thư mục `data/`.
2. **Chunking:** Cắt văn bản thành các đoạn nhỏ 600-800 ký tự với độ chồng lấp (overlap) phù hợp.
3. **Embedding & Indexing:** Chuyển đổi văn bản thành vector và lưu vào `faiss_index`.
4. **Retrieval:** Tìm kiếm 5 đoạn thông tin liên quan nhất khi người dùng đặt câu hỏi.
5. **Generation:** Kết hợp ngữ cảnh (Context) và câu hỏi (Question) vào Prompt chuẩn để Gemini 1.5 tạo câu trả lời.

## Cấu trúc thư mục
```text
tech-chatbot-rag-multiagent/
├── app/
│   ├── core/           # Cấu hình API Key, Embeddings
│   ├── rag/            # Logic chính (Loader, Chunking, VectorStore, Chains)
│   └──main.py             # File điều hướng chính (Entry Point)
├── skills/          # Các kỹ năng mở rộng (Search, Compare, Calculator)
├── prompts/            # Chứa file rag_prompt.txt (System Prompt)
├── data/               # Thư mục chứa dữ liệu đầu vào (.csv, .pdf)
├── faiss_index/        # Cơ sở dữ liệu Vector (tự sinh ra sau khi build)
├── .env                # Biến môi trường (API Key)
└── requirements.txt    # Danh sách thư viện cần thiết
```

## Cài đặt & Sử dụng

### 1. Cài đặt môi trường
```bash
# Tạo môi trường ảo
python -m venv venv
source venv/Scripts/activate  # Trên Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Cấu hình API Key
Tạo file `.env` hoặc file `app/core/config.py` và thêm khóa Gemini của bạn:
```python
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

### 3. Chạy ứng dụng
```bash
python main.py
```
*Hệ thống sẽ tự động kiểm tra, nếu chưa có database, nó sẽ tiến hành xây dựng từ dữ liệu trong thư mục `data/`.*

## Nguyên tắc hoạt động (Guardrails)
* Chatbot tuyệt đối không "chém gió" nếu không tìm thấy dữ liệu trong `context`.
* Không tìm kiếm thông tin bên ngoài nếu không được phép.
* Luôn ưu tiên độ chính xác về thông số kỹ thuật (Chip, RAM, Giá).

## Định hướng phát triển (Roadmap)
- [ ] Tích hợp **Multi-Agent** (LangGraph/CrewAI) để phân chia vai trò tư vấn.
- [ ] Thêm kỹ năng tìm kiếm Web thời gian thực (Search Skill) khi dữ liệu CSV cũ.
- [ ] Xây dựng giao diện Web (Streamlit hoặc Next.js).
