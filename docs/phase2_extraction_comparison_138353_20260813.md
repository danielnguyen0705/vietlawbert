# Đối chứng extraction record 138353 — 2026-08-13

## Mục tiêu

So sánh text layer hiện tại của record `138353` với OCR đối chứng để xác định liệu có thể hạ warning corpus review bằng evidence kỹ thuật hay không.

## Nguồn kiểm tra

- `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/checkpoint.jsonl`
- PDF gốc tải lại từ attachment của record `138353`
- PDF diagnostic:
  - `outputs/pdf_diagnostics_20260813/138353.pdf`
  - `outputs/pdf_diagnostics_20260813/138353_forced_ocr_pages1_5.txt`

## Kết quả inventory PDF

- `Pages: 59`
- `File size: 10966911 bytes`
- Producer: `3-Heights(TM) PDF Optimization Shell 4.8.25.2`
- Có font trong `pdffonts`, nhưng đa số font `emb: no`
- Có ít nhất một font `Courier` với `encoding = Custom`

Điểm này rất quan trọng: PDF **có text layer**, nhưng text layer không đáng tin cậy vì mapping font/encoding có dấu hiệu lỗi. Đây là mẫu kinh điển dẫn tới mojibake dù `pdftotext`/text-layer extraction vẫn trả nhiều ký tự.

## Text layer hiện tại trong pipeline

Record hiện tại có:

- `status = HTML_VALID`
- `extraction_method = PDF_TEXT_LAYER`
- warnings:
  - `CONTENT_RECOVERED_FROM_PDF`
  - `RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED`
  - `RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED`
  - `RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED`

Metrics của text layer:

- `total_chars = 98337`
- `suspect_token_count = 1899`
- `suspect_char_density = 0.019311144330211416`
- `vietnamese_char_ratio = 0.0005700673901093307`
- `structure_marker_count = 224`
- `trailing_fragment_suspected = False`

Quan sát trực tiếp:

- mở đầu bị lỗi nặng: `CQNG 110A XA 1101 CHU NGHIA VIVI' NAM`
- xuất hiện dày đặc `(cid:...)`
- cuối tài liệu nhiễu gần như không dùng được cho corpus
- dù vẫn nhận ra được chủ đề Thông tư chứng khoán, chất lượng ký tự quá thấp để tin cậy cho RAG/training

## OCR đối chứng

Đã OCR đối chứng 5 trang đầu ở 150 DPI. Kết quả đầu ra tốt hơn text layer rõ rệt:

- quốc hiệu đọc được gần đúng: `CONG HOA XA HOI CHU NGHIA VIET NAM`
- tiêu đề nhận diện được: `THONG TU`
- các câu căn cứ và số hiệu còn đọc được
- không còn `(cid:...)`

Metrics OCR đối chứng cho 5 trang đầu:

- `total_chars = 13057`
- `suspect_token_count = 0`
- `suspect_char_density = 0.0`
- `vietnamese_char_ratio = 0.049225546361128796`
- `structure_marker_count = 78`
- `trailing_fragment_suspected = False`

Điểm này cho thấy **OCR đối chứng tốt hơn text layer hiện tại rất nhiều về khả năng đọc**.

## Kết luận kỹ thuật

`138353` không phải case “text layer tốt nhưng hơi bẩn”. Đây là case text layer bị lỗi encoding có hệ thống. OCR đối chứng chứng minh có khả năng thu được text sạch hơn đáng kể.

Tuy vậy, trong vòng xử lý hiện tại vẫn còn 2 giới hạn:

1. OCR đối chứng mới chạy trên 5 trang đầu, chưa phải toàn bộ 59 trang.
2. Pipeline hiện tại chưa có rule tự động “nếu text layer nhiễu nặng thì ưu tiên OCR thay thế toàn bộ”.

Vì vậy **chưa thể gỡ các warning corpus review của `138353` ngay bây giờ**.

## Quyết định

- Giữ `138353` ở trạng thái blocked khỏi corpus.
- Không dùng text layer hiện tại làm corpus-ready evidence.
- Không sửa đè raw `content_html` hiện tại.
- Nếu muốn cứu record này, cần một bước mới: OCR toàn bộ PDF rồi so sánh định lượng với text layer để chọn extraction tốt hơn theo quality score.

## Khuyến nghị bước tiếp theo

1. Thêm chế độ fallback ưu tiên OCR khi text layer có `suspect_char_density` cao và `vietnamese_char_ratio` rất thấp.
2. OCR toàn bộ 59 trang của `138353`.
3. So sánh hai bản bằng metrics thống nhất, rồi mới cân nhắc promote OCR output thành candidate text song song với raw evidence.

## Kết luận ngắn

`138353` chưa xử lý xong. Nhưng đã có bằng chứng mạnh rằng hướng đúng không phải cố cứu text layer hiện tại, mà là OCR đối chứng/toàn phần.
