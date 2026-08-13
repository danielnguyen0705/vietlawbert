# Audit thủ công record 133328 — 2026-08-13

## Mục tiêu

Đánh giá recovered OCR record `133328` để xác định liệu warning `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED` có thể được hạ xuống hay chưa, và liệu record này đã đủ corpus-ready cho Phase 2 hay chưa.

## Nguồn kiểm tra

- `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/checkpoint.jsonl`
- record `document_id = 133328`
- text recovered từ `PDF_OCR`
- metadata chính:
  - `docNum = 11/2001/NQ-HĐND`
  - `official_group_code = VBQPPL`
  - `official_form_code = NQ`
  - `pdf_url = https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/minio/buckets/vbpl/133328/VanBanGoc_11.2001.pdf/download`

## Quan sát thực tế

Record có `status = HTML_VALID`, `content_chars = 13716`, warnings chỉ còn:

- `CONTENT_RECOVERED_FROM_PDF`
- `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED`

Metrics recovered text:

- `total_chars = 13716`
- `suspect_token_count = 0`
- `suspect_char_density = 0.0`
- `vietnamese_char_ratio = 0.07050686487044493`
- `structure_marker_count = 21`
- `trailing_fragment_suspected = False`

Normalization metadata:

- `method = PDF_OCR`
- `flags = ["whitespace_normalized"]`
- `normalized_chars = 13715`

## Dấu hiệu cho thấy OCR tương đối đầy đủ

Text recovered có đủ các thành phần pháp lý lớn, theo đúng thứ tự thường gặp của một nghị quyết HĐND:

- header cơ quan ban hành: `HOI DONG NHAN DAN`
- quốc hiệu: `CONG HOA XA HOI CHU NGHIA VIET NAM`
- số hiệu và ngày tháng
- thông tin kỳ họp: `KHOA X, KY HOP THU 4`
- tiêu đề `NGHI QUYET`
- phần căn cứ
- phần `QUYET NGHI`
- mục đánh giá kết quả năm trước
- mục mục tiêu, chỉ tiêu năm 2001
- mục nhiệm vụ, giải pháp chính
- mục tổ chức thực hiện
- đoạn kết thông qua nghị quyết
- phần `Nơi nhận` / phân phối cuối văn bản

Nội dung thân bài không chỉ có mở đầu mà còn kéo dài qua nhiều mục, gồm cả các chỉ tiêu định lượng như GDP, sản lượng lương thực, diện tích rừng trồng mới, ngân sách, việc làm, tỷ lệ hộ đói nghèo. Phần cuối vẫn còn các mục về quốc phòng, an ninh, tuyên truyền giáo dục pháp luật, phong trào thi đua, và `Tổ chức thực hiện`.

Điểm này quan trọng vì cho thấy record không phải dạng OCR ngắn hoặc rơi mất toàn bộ nửa sau văn bản như `68560`.

## Dấu hiệu chất lượng còn hạn chế

Dù cấu trúc khá đầy đủ, OCR vẫn còn lỗi nhận dạng rõ rệt:

- lỗi dấu tiếng Việt: `Thdi`, `phic`, `Nguyén`, `Diéu`, `Nuéc`, `cha`, `két qua`, `x4 héi`
- lỗi ký tự/cụm từ đầu văn bản: `EECA ais al fy he kee yA Pe ne a es ed ee een ei pene fe`
- lỗi ở số hiệu: `Sé:4/  /2001/NQ-HDND`
- lỗi ở phần ký/đóng văn bản cuối: `CHU TICH HPND TINH THAI NGUYEN`
- nhiều chỗ chữ bị thay thế hoặc dính ký tự lạ như `ttte`, `phat trién`, `céng`, `d6i`, `6`, `1a`

Tuy không có `(cid:...)` hay replacement char, đây vẫn là OCR text chứ chưa phải bản corpus sạch. Một số token số liệu và ngày tháng cần đối chiếu trực tiếp với PDF gốc nếu muốn đưa vào training/RAG chất lượng cao.

## Đánh giá completeness

Theo evidence hiện có, record `133328` có khả năng cao là gần đầy đủ về cấu trúc và phần lớn nội dung chính. Không thấy tín hiệu mạnh của:

- thiếu nửa cuối văn bản
- cắt đột ngột cuối file
- vỡ encoding nặng
- text quá ngắn so với một nghị quyết nhiều mục

Tuy nhiên, chưa có bằng chứng page-level trực tiếp từ PDF gốc trong vòng audit này. Vì vậy chưa thể nâng từ “OCR review required” lên “corpus-ready” chỉ từ recovered text hiện có.

## Kết luận cho gate Phase 2

`133328` là recovered record tốt nhất trong 3 record blocked hiện tại.

Tuy vậy, **chưa nên gỡ warning `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED` ở thời điểm này**. Lý do:

1. nguồn vẫn là `PDF_OCR`, chưa phải text layer sạch;
2. còn nhiều lỗi OCR tác động tới dấu, từ, số hiệu và một phần entity cuối văn bản;
3. chưa có đối chiếu page-level hoặc text-level với PDF gốc để xác nhận độ trung thực đủ cho corpus training/RAG.

## Quyết định thực tế

- Giữ `HTML_VALID` ở tầng technical retrieval.
- Giữ `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED` ở tầng corpus review.
- Không xếp `133328` vào corpus-ready tự động.
- Có thể ưu tiên `133328` làm ứng viên review thủ công đầu tiên nếu muốn thử xây quy trình “promote after manual approval”.

## Khuyến nghị bước kế tiếp riêng cho 133328

1. Đọc trực tiếp PDF gốc theo page để đối chiếu số trang, phần mở đầu và phần kết.
2. So khớp ít nhất các trường: số hiệu, ngày ban hành, tên cơ quan, tiêu đề nghị quyết, các mục lớn, đoạn thông qua nghị quyết, phần nơi nhận.
3. Nếu đối chiếu đạt, có thể xem xét cơ chế `manual_review_passed` thay vì gỡ warning bằng heuristic tự động.

## Kết luận ngắn

`133328` đủ tốt để giữ trong review queue ưu tiên cao. Chưa đủ bằng chứng để coi là corpus-ready tự động. Warning OCR review nên giữ nguyên.
