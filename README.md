# News Reader AI 🚀
Hệ thống AI Agent dựa trên cấu trúc Agentic RAG để thu thập dữ liệu tin tức mới nhất ở Việt Nam (nguồn từ Vnexpress.net). Xây dựng dựa trên thư viện langchain, langgraph và qdrant. Có thể triển khai giao diện Streamlit để tương tác hoặc hỏi đáp thông qua Discord bot.

## Tính năng 
- **Thu thập dữ liệu**: Tự động lấy tin tức mới nhất theo danh mục (Thể thao, Khoa học, Công nghệ, ...).
- **AI Agent**: Quy trình xử lý tin tức: lọc nội dung, tóm tắt và trả lời câu hỏi dựa trên ngữ cảnh.
- **Vector Database**: Sử dụng Vector Database để truy xuất thông tin từ kho dữ liệu tin tức đã lưu trữ.
- **Triển khai**: 
    - **Discord Bot**: Tương tác trực tiếp qua lệnh `/`.
    - **Streamlit Web App**: Giao diện trực quan để chọn chuyên mục và đọc tin.

## Công nghệ sử dụng
- **Ngôn ngữ**: Python 3.10+
- **LLM Framework**: LangChain, LangGraph
- **Database**: Qdrant
- **Interface**: Discord.py, Streamlit
- **Scraping**: BeautifulSoup4, Requests

## Hướng dẫn nhanh

### 1. Cài đặt môi trường
```bash
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cấu hình
Tạo file `.env` và thêm các khóa API cần thiết (OpenAI/Google API Key, Discord Token).

### 3. Chạy ứng dụng
- **Chạy Discord Bot**:
  ```bash
  python discord_bot.py
  ```
- **Chạy Web App**:
  ```bash
  streamlit run app.py
  ```
  
Để chạy được Discord Bot, bạn cần thực hiện các bước sau:
1. Truy cập [Discord Developer Portal](https://discord.com/developers/applications).
2. Nhấn **New Application** và đặt tên cho Bot của bạn.
3. Đến mục **Bot** ở thanh menu bên trái:
    - Nhấn **Reset Token** để lấy `DISCORD_TOKEN`. Lưu token này vào file `.env`.
    - Tại mục **Privileged Gateway Intents**, bật **Message Content Intent** để Bot có thể đọc được tin nhắn.
4. Đến mục **OAuth2** -> **URL Generator**:
    - Chọn scope: `bot`, `applications.commands`.
    - Chọn permissions: chọn các quyền cụ thể như `Send Messages`, `Read Message History`.
    - Copy đường link được tạo và dán vào trình duyệt để mời Bot vào Server của bạn.

### 4. Lưu ý 
Chương trình chạy mặc định mô hình local llm: qwen3-4b-2507 và embeddings: halong_embedding thông qua chương trình LM-Studio. Chạy ổn định với RTX 3060 12GB. Có thể sử dụng các API OpenAI hoặc Gemini nếu không muốn chạy local.