# Kiểm chứng Chainlit + FastAPI — 12/09/2026

Nhánh: `ubuntu-feature/interface`. Phạm vi: demo mock và adapter RAG; chưa nghiệm thu dữ liệu pháp luật thật.

## Kiểm thử tự động

Python 3.12.14 trên Windows, môi trường `.venv-chat` chỉ cài bộ dependency chat/dev:

```text
python -m pytest tests/unit/test_chat_api.py tests/unit/test_chat_ui.py -q
27 passed
```

Bao phủ validation, ba chủ đề mock, không có căn cứ, không import AI trong mock, lỗi 500/502/503, lifecycle provider, adapter generator, metadata thiếu, RAG bận nhưng health vẫn đáp ứng, HTTP timeout/kết nối/lỗi payload, định dạng footer, tên nguồn theo lượt và callback kết thúc loading.

Có ba cảnh báo deprecation từ dependency (Starlette/AnyIO/Traceloop), không có test thất bại. `uv pip check` xác nhận dependency tương thích; `git diff --check` không có lỗi whitespace.

## Kiểm chứng thực tế

| Hạng mục | Windows local | Docker Linux |
| --- | --- | --- |
| Khởi động UI/API trên 8001/8000 | Đạt | Đạt; cả hai container healthy |
| Câu hỏi demo, câu trả lời và footer | Đạt | Đạt |
| Bấm nguồn và đọc số hiệu/nội dung/điểm | Đạt | Đạt |
| Đổi top_k từ 3 thành 1 | Đạt | Đạt |
| Câu ngoài mẫu trả không có căn cứ | Đạt | Đạt |
| Dừng API: báo lỗi, kết thúc loading | Đạt | Đạt |
| API phục hồi: gửi lại thành công | — | Đạt sau khi API sẵn sàng |
| Hai phiên tách lịch sử và top_k | — | Đạt: phiên mới giữ 3 khi phiên cũ chọn 1 |
| Giới thiệu tiếng Việt và nhãn giả lập | Đạt | Đạt, nội dung Markdown được đóng gói |

Docker được kiểm thử bằng Linux containers trên Docker Desktop; chưa chạy trên một máy chủ Ubuntu riêng. Compose chuẩn kèm hướng dẫn dùng được cho môi trường Docker trên Ubuntu.

## Bằng chứng môi trường Docker

- Build thành công với `Dockerfile.chat`, Python 3.12 và `uv` 0.12.13; cài phiên bản từ `requirements-chat.txt`.
- Project riêng `vietlawbert-chat` gồm `api` và `ui`, bind cổng trên `127.0.0.1`.
- Trong image API: không có package `torch` hoặc `rag`; mock vẫn ready và trả kết quả.
- Health trả `mode=mock`, `ready=true`, model và database là `not_required`.
- Log UI xác nhận gọi `http://api:8000/api/v1/chat` và nhận HTTP 200.
- OpenAPI đang chạy có ví dụ ChatRequest/ChatResponse và khai báo mã 200, 422, 500, 502, 503.

## Giới hạn nghiệm thu

Adapter RAG được kiểm thử bằng generator giả lập. Kết nối database thật, dữ liệu đã ingest, tải model và chất lượng trả lời pháp luật cần kiểm chứng khi dữ liệu sẵn sàng. Attribution hiện là đối sánh trích dẫn, không phải thước đo độ đúng pháp lý.
