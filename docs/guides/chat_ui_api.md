# Chatbot Chainlit + FastAPI

Luồng: trình duyệt → Chainlit (8001) → HTTP JSON → FastAPI (8000) → provider mock/RAG.
Mỗi câu hỏi độc lập. Không có streaming, upload, đăng nhập hay lưu lịch sử lâu dài.

## Chạy local Windows (PowerShell)

Từ thư mục gốc repo, dùng Python 3.12:

```powershell
py -3.12 -m venv .venv-chat
.\.venv-chat\Scripts\python.exe -m pip install -r requirements-chat.txt
```

Terminal 1, từ thư mục gốc:

```powershell
$env:CHAT_MODE = 'mock'
.\.venv-chat\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1 --log-level info
```

Terminal 2, từ thư mục gốc:

```powershell
$env:CHAT_API_URL = 'http://localhost:8000'
Set-Location ui
..\.venv-chat\Scripts\python.exe -m chainlit run chainlit_app.py --host 127.0.0.1 --port 8001 --headless
```

Mở UI tại http://localhost:8001, trang giới thiệu qua nút Readme, Swagger tại http://localhost:8000/docs.
Chạy Chainlit từ `ui/` để đọc đúng `chainlit.md` và `.chainlit/config.toml`.

## Chạy Ubuntu

```bash
python3.12 -m venv .venv-chat
.venv-chat/bin/python -m pip install -r requirements-chat.txt
CHAT_MODE=mock .venv-chat/bin/python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1
# Terminal thứ hai, từ thư mục gốc:
cd ui
CHAT_API_URL=http://localhost:8000 ../.venv-chat/bin/python -m chainlit run chainlit_app.py --host 127.0.0.1 --port 8001 --headless
```

## Docker độc lập

```bash
docker compose -f docker-compose.chat.yml up --build -d --wait --wait-timeout 60
docker compose -f docker-compose.chat.yml ps
docker compose -f docker-compose.chat.yml logs --tail=50 api ui
docker compose -f docker-compose.chat.yml down
```

Docker Engine phải đang chạy. Compose này dùng project riêng `vietlawbert-chat`, chỉ tạo UI/API, không chạy database hoặc tải mô hình.
Cổng chỉ bind localhost. UI trong container gọi `http://api:8000`.
Compose gắn toàn bộ thư mục repo trên máy vào `/app` của cả API và UI (bind mount).
Trong WSL, chạy từ `/mnt/d/vietlawbert` để dùng cùng repo `D:\vietlawbert`.
Các file như `main.py`, `pipeline/`, `api/` và `ui/` trong container luôn phản ánh
file hiện tại trên máy, kể cả thay đổi chưa commit. Ghi file trong `/app` cũng ghi
vào repo trên máy. Image độc lập chỉ đóng gói API/UI; toàn bộ repo được cung cấp
bởi bind mount khi chạy Compose.

Sau lần đầu đổi cấu hình mount, tạo lại container:

```bash
docker compose -f docker-compose.chat.yml up -d --build --force-recreate --wait
```

Sau mỗi lần pull/merge hoặc sửa code, khởi động lại tiến trình để bỏ các module
Python đã nạp trong bộ nhớ (file trên đĩa đã đồng bộ, tiến trình không tự reload):

```bash
docker compose -f docker-compose.chat.yml restart
docker compose -f docker-compose.chat.yml ps
```

Nếu đổi dependency `requirements-chat.txt` hoặc Dockerfile, chạy lại lệnh
`up -d --build --force-recreate --wait` thay vì chỉ restart.

Image nhẹ chỉ cài dependency chat, không chứa RAG/AI dependencies. Việc có toàn bộ
source trong `/app` không tự chạy `main.py` hay bật pipeline: Compose vẫn chạy
API/UI với `CHAT_MODE=mock`. Muốn chạy RAG thật cần cài thêm dependency và cấu hình
database/model như mục bên dưới. Môi trường `.venv-chat` trên Windows không dùng
để chạy Python trong container Linux.
Image dùng `uv` phiên bản cố định để cài cùng bộ dependency đã khóa. Dừng hai tiến trình local trước khi chạy Docker vì cả hai cách dùng cổng 8000/8001.

## Hợp đồng

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/chat -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes('{"query":"Demo giao thông","top_k":2,"stream":false}'))
```

- Query được trim, 1–4.000 ký tự; top_k là số nguyên 1–10, mặc định 3. Mỗi chủ đề mock có tối đa 3 căn cứ dù top_k lớn hơn.
- `stream=true` trả 422. Không có căn cứ vẫn trả 200 với danh sách rỗng.
- Kết quả gồm answer, contexts, has_context, attribution_score, latency_ms, model_used, mode và query.
- Metadata thiếu là null. Attribution 0–1 là đối sánh trích dẫn, không phải độ chính xác pháp lý; UI đổi sang phần trăm.
- Lỗi 422: đầu vào sai; 503: provider chưa sẵn sàng/đang bận; 502: LLM thất bại; 500: lỗi nội bộ.
- Health 200 khi provider sẵn sàng, 503 khi khởi tạo thất bại. Mock báo database/model `not_required`; RAG báo database `not_checked` vì endpoint không thực hiện probe riêng.

## Kịch bản demo

1. Gửi “Demo giao thông”; kiểm tra nhãn mock, câu trả lời, ba nguồn và footer.
2. Bấm từng nguồn; kiểm tra nội dung giả lập cùng rerank/graph boost.
3. Đổi số căn cứ thành 1; gửi “Demo doanh nghiệp”; chỉ có một nguồn, nguồn cũ vẫn thuộc câu trước.
4. Gửi “Demo hành chính”, rồi “xin chào”; câu ngoài mẫu không có căn cứ.
5. Mở hai phiên trình duyệt và gửi câu khác nhau; mỗi phiên hiển thị kết quả riêng.
6. Dừng API và gửi câu hỏi; UI kết thúc loading, báo lỗi kết nối. Khởi động lại API và gửi lại.

## Nối RAG thật sau này

Cài dependency của pipeline bằng `requirements.txt` bên cạnh bộ dependency chat; kiểm tra xung đột trước khi dùng chung môi trường. Cấu hình database, model và provider LLM theo hướng dẫn RAG hiện có rồi đặt `CHAT_MODE=rag` và khởi động lại FastAPI.

Adapter lazy-import `LegalGenerator`, gọi `ask(query, top_k)` và `close()`; UI không cần sửa.
Một process chứa một generator; chỉ chạy **1 worker**. Request RAG đang bận trả 503 cho request kế tiếp, health vẫn đáp ứng. Lỗi nạp model/database giữ API ở trạng thái unavailable, không âm thầm trả mock.

Phiên bản này chỉ kiểm thử adapter bằng generator giả lập. Chất lượng truy xuất, kết nối database thực tế và câu trả lời pháp luật phải nghiệm thu riêng khi dữ liệu sẵn sàng. Health không xác nhận kho dữ liệu đã được nạp hay tất cả database đang khỏe.

## Cấu hình

Biến môi trường ưu tiên hơn `.env` ở thư mục gốc:

| Biến | Mặc định | Nơi dùng |
| --- | --- | --- |
| CHAT_MODE | mock | API: mock hoặc rag |
| CHAT_API_URL | http://localhost:8000 | UI: địa chỉ API, không gồm `/api/v1` |
| CHAT_TIMEOUT_SECONDS | 300 | UI: timeout phản hồi; connect timeout 5 giây |

Timeout UI không hủy tác vụ RAG đồng bộ đã chạy ở backend; gửi lại trong lúc đó có thể nhận 503. Client không tự retry.

## Kiểm thử

```powershell
.\.venv-chat\Scripts\python.exe -m pip install -r requirements-chat-dev.txt
.\.venv-chat\Scripts\python.exe -m pytest tests/unit/test_chat_api.py tests/unit/test_chat_ui.py -q
```

Không chạy toàn bộ test pipeline khi chỉ kiểm tra chat. Bộ test chat không cần GPU, LLM, model hay database.
Kết quả nghiệm thu local và Docker: [Báo cáo kiểm chứng](../reports/chat_ui_api_validation.md).
Khi cập nhật dependency, phân giải universal trên Python 3.12 từ `requirements-chat.in` bằng `uv pip compile --universal`, chạy lại test và smoke UI trước khi commit lock mới.
