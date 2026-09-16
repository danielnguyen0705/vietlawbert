# Kiểm tra merge ubuntu vào ubuntu-feature/interface

Ngày kiểm tra: 2026-09-16.

Nguồn merge: `origin/ubuntu` tại `c5aea93`; nhánh giao diện trước merge tại `6bf2392`.

## Nội dung kết hợp

- Dùng `merge.renormalize=true` và `-Xignore-space-at-eol`; 40 xung đột trong mô phỏng thông thường giảm còn 3 file cần kết hợp thủ công.
- Thêm `.gitattributes`: file văn bản dùng LF; `.bat` và `.cmd` dùng CRLF. Chuẩn hóa cả nội dung Git và file làm việc.
- `rag/generator.py`: nhận import từ ubuntu; giữ cấu hình timeout/local LLM, xử lý phản hồi rỗng và chuẩn hóa trích dẫn Unicode từ nhánh giao diện.
- `rag/retriever.py`: nhận truy vấn Elasticsearch `query=...`, `best_fields`, `tie_breaker`, bộ lọc hiệu lực và các cập nhật truy xuất từ ubuntu.
- `pipeline/ingest_pipeline.py`: nhận xử lý `effStatus` dạng số hoặc đối tượng; giữ `skip_existing`, định danh cho điều khoản trùng số, kiểm tra đủ batch và truyền lỗi để tránh ghi checkpoint thành công giả. Trạng thái thiếu dữ liệu vẫn là `Chưa xác định`.
- `database/qdrant_client.py`: nhận xử lý vector NumPy và chuẩn hóa UUID; giữ `wait=True` để xác nhận ghi dữ liệu.
- Giữ phần API/UI/demo. Bổ sung hàm `pending_ids` tương thích với người gọi OCR cũ.
- Sửa regex dẫn chiếu để giữ chữ Đ trong số hiệu NĐ-CP; chỉnh kỳ vọng kiểm thử AST theo nhãn chương có tiêu đề mà parser đã xuất từ trước merge.

## Kết quả kiểm thử

```powershell
.venv-chat\Scripts\python.exe -m pytest tests/unit/test_chat_api.py tests/unit/test_chat_ui.py -q
# 27 passed

.\.venv-merge\Scripts\python.exe -m pytest tests/unit/test_demo_pipeline.py tests/unit/test_merge_ubuntu.py tests/unit/test_ast_parser.py tests/unit/test_text_cleaner.py tests/unit/test_legal_chunker.py tests/unit/test_crawl_audit.py tests/unit/test_mrl_loss.py -q
# 32 passed
```

Kiểm thử hồi quy mới bao gồm trạng thái hiệu lực thiếu/dạng số, client Elasticsearch trực tiếp, định danh Qdrant ổn định, vector NumPy, phân biệt Điều 1 với Điều 10 và Qdrant trong bộ nhớ: nhập lại không tăng số điểm, kiểm tra văn bản tồn tại và lọc hiệu lực khi tìm kiếm.

Tất cả file Python được Git theo dõi qua kiểm tra cú pháp. Không còn file xung đột; `git diff --cached --check` đạt; nội dung staging không còn CRLF hoặc xuống dòng hỗn hợp.

## Giới hạn

59 kiểm thử nêu trên đạt. Chưa chạy toàn bộ kiểm thử integration, crawl/OCR thực tế, Elasticsearch server hoặc chuỗi demo với mô hình embedding và dịch vụ LLM thật. Không có container dữ liệu đang chạy lúc kiểm tra. `.venv-merge` là môi trường kiểm thử riêng dùng thư viện nền của Python hệ thống và các thư viện bổ sung, được bỏ qua trong Git.
