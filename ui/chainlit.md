# VietLawBERT

Hệ thống Tra cứu & Hỏi đáp Pháp luật Thông minh.

**Chế độ demo — dữ liệu minh họa** là chế độ mặc định khi chưa có kho dữ liệu.
Các văn bản mang mã `DEMO-*` và điểm số đi kèm đều là giả lập, không phải luật hiện hành.
Chế độ thực tế được thông báo khi bắt đầu chat và trên từng câu trả lời.

## Bắt đầu

- Demo giao thông
- Demo doanh nghiệp
- Demo hành chính

Chọn số căn cứ trong cài đặt chat. Bấm tên nguồn dưới câu trả lời để mở nội dung chi tiết.
Mỗi câu hỏi độc lập; nội dung lượt trước không được gửi làm ngữ cảnh.

Khi kết nối dữ liệu thật, backend sử dụng adapter cho động cơ RAG hiện có.
Attribution là chỉ số đối sánh trích dẫn, không phải độ chính xác pháp lý.
