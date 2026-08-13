# Completeness audit record 68560 — 2026-08-13

## Mục tiêu

Kiểm tra record `68560` để xác định liệu OCR ngắn hiện tại có phải do thiếu nội dung/page hay không, và liệu record này có thể được coi là corpus-ready hay chưa.

## Nguồn kiểm tra

- `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/checkpoint.jsonl`
- PDF gốc tải lại từ attachment của record `68560`
- PDF diagnostic:
  - `outputs/pdf_diagnostics_20260813/68560.pdf`
  - `outputs/pdf_diagnostics_20260813/68560_forced_ocr_300dpi.txt`

## Kết quả inventory PDF

- `Pages: 2`
- `File size: 66983 bytes`
- Title: `Microsoft Word - NQ16-2002.doc`
- Có font `TimesNewRomanPSMT`
- `pdftotext -layout` gần như không trả text dùng được

Điểm quan trọng: đây là PDF chỉ có **2 trang**. Vì vậy OCR output ngắn không nhất thiết là thiếu nội dung; có thể văn bản gốc thật sự ngắn.

## Record hiện tại trong pipeline

- `status = HTML_VALID`
- `extraction_method = PDF_OCR`
- warnings:
  - `CONTENT_RECOVERED_FROM_PDF`
  - `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED`
  - `OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED`
  - `RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED`
  - `RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED`

Metrics hiện tại:

- `total_chars = 2056`
- `suspect_token_count = 1`
- `suspect_char_density = 0.00048638132295719845`
- `vietnamese_char_ratio = 0.04926470588235294`
- `structure_marker_count = 3`
- `trailing_fragment_suspected = False`

## OCR đối chứng 300 DPI

Đã OCR lại 2 trang PDF ở 300 DPI. Kết quả xác nhận:

- có tiêu đề, số hiệu, ngày tháng, căn cứ, phần `QUYET NGHI`
- có danh sách mức huy động theo vùng
- có mục đối tượng được miễn huy động
- có mục giao `UBND` tổ chức thực hiện
- có phần ký cuối văn bản `TM. HOLDONG NHAN DAN TINH DAK LAK`

Nghĩa là OCR hiện tại **không bị mất toàn bộ nửa sau do thiếu trang**. Nội dung cuối văn bản đã có mặt. Vấn đề chính là **chất lượng OCR còn thấp**, không phải thiếu trang rõ rệt.

## Chất lượng còn hạn chế

Dù completeness theo page có vẻ ổn, chất lượng vẫn chưa đủ sạch:

- số hiệu lỗi: `16/2002/NO-HD`
- token rác đầu trang: `HIP CR`
- lỗi ký tự cuối văn bản: `HPND tnh Duk Lak`, `qua nea 11 thane 7 ndin 2002`
- vẫn có nhiều lỗi dấu và từ
- `pdftotext` không cứu được vì text layer gần như không usable

Vì vậy record này không còn là case “nghi thiếu trang” mạnh như trước, nhưng vẫn là case “OCR chất lượng chưa đạt corpus-ready”.

## Kết luận

`68560` có khả năng **đủ 2 trang gốc**, tức completeness theo page tạm ổn hơn nghi ngờ ban đầu. Tuy nhiên record vẫn chưa nên vào corpus vì:

1. OCR còn ngắn so với ngưỡng review.
2. còn lỗi nhận dạng ở số hiệu, ngày tháng, chức danh và nhiều token.
3. không có text layer sạch để đối chiếu.

Do đó **giữ nguyên toàn bộ warning review hiện tại** là hợp lý.

## Quyết định

- Không coi `68560` là corpus-ready.
- Giữ `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED`.
- Giữ `OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED`.
- Giữ các warning chất lượng OCR khác.

## Kết luận ngắn

`68560` không cho thấy thiếu trang rõ rệt nữa, nhưng vẫn là OCR output chất lượng thấp. Record này vẫn block corpus gate.
