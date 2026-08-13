# Audit chất lượng content recovered Phase 2

Ngày audit: 2026-08-13

## Phạm vi và nguồn dữ liệu

Audit kiểm tra 3/3 record có cảnh báo `CONTENT_RECOVERED_FROM_PDF` trong pilot 120:

- `138353`: trích xuất từ PDF text layer.
- `133328`: trích xuất bằng OCR.
- `68560`: trích xuất bằng OCR.

Nguồn trực tiếp: `law_dataset/artifacts/vbpl_pilot_120_phase2_fallback_20260813/documents.jsonl.gz` và `checkpoint.jsonl`. Đối chiếu số lượng dùng `report.json` và `REPORT.md` cùng thư mục.

Audit này đánh giá technical readiness của text cho legal-structure parsing/RAG. Không kết luận hiệu lực hoặc giá trị pháp lý của văn bản.

## Kết quả tổng hợp

| ID | Phương thức | Số ký tự | Confidence | Đánh giá technical readiness |
|---|---|---:|---:|---|
| `138353` | `PDF_TEXT_LAYER` | 98.337 | 0,95 | Không đạt pass corpus; cần review/re-extraction |
| `133328` | `PDF_OCR` | 13.716 | 0,70 | Có thể giữ review queue; chưa được đưa thẳng vào corpus/RAG |
| `68560` | `PDF_OCR` | 2.056 | 0,70 | Không đạt corpus gate; chỉ giữ review/metadata hoặc re-OCR |

Pilot report ghi nhận 3 record recovered, trong đó 2 record OCR; `pdf_fetch_failed = 0` và `pdf_extraction_insufficient = 2` trên 120 record.

## Nhận xét từng record

### ID `138353`

Metadata cho biết đây là `47/VBHN-BTC`, nhóm chính thức `VBHN`, hình thức `VBHN`, tiêu đề liên quan đến hướng dẫn giám sát tuân thủ trong lĩnh vực chứng khoán. Text layer tạo ra 98.337 ký tự và có nhiều marker cấu trúc như khoảng 45 lần `Điều`, 176 lần `Khoản`, và 10 lần `Căn cứ` theo phép đếm heuristic trên raw text.

Tuy nhiên, nội dung quan sát được chứa lỗi mã hóa/OCR-like nghiêm trọng dù method là text layer: `CQNG 110A XA 1101`, `VIVI' NAM`, nhiều chuỗi `(cid:...)`, ký tự điều khiển trình bày, và phần cuối bị nhiễu nặng. Audit đếm được khoảng 1.915 marker nghi ngờ mojibake trong text raw. Vì vậy `extraction_confidence = 0.95` đang phản ánh việc text layer đủ dài, chưa phản ánh chất lượng ký tự hoặc khả năng parse.

Kết luận: không được coi là corpus-ready chỉ vì đạt `MIN_PDF_TEXT_CHARS`. Cần gắn review warning chất lượng extraction, thử pipeline làm sạch/mapping encoding, hoặc chạy OCR có chọn lọc để so sánh. Giữ text raw bất biến để evidence; không ghi đè bằng text đã sửa.

### ID `133328`

Metadata cho biết đây là `11/2001/NQ-HĐND`, nhóm `VBQPPL`, hình thức `NQ`, về nhiệm vụ năm 2001 của tỉnh Thái Nguyên. OCR text có 13.716 ký tự, chứa phần đầu văn bản, căn cứ, mục `QUYẾT NGHỊ`, các đề mục và phần `Nơi nhận` ở cuối. Heuristic đếm được khoảng 6 marker `Điều`, 5 marker `Chương`, 1 marker `Khoản`, và 1 marker `QUYẾT NGHỊ`.

Text vẫn có lỗi OCR phổ biến: mất/đổi dấu tiếng Việt, ví dụ `Nguyén`, `néu`, `duoc`, `Thdéi`, cùng một số ký tự lạ. Dù vậy, cấu trúc tổng thể và nhiều nội dung chính còn nhận diện được.

Kết luận: có giá trị trong review queue và có thể dùng làm mẫu đánh giá OCR, nhưng chưa đủ cơ sở đưa thẳng vào corpus huấn luyện hoặc RAG. Cần kiểm tra page-level completeness, số trang, tỷ lệ ký tự lỗi, và đối chiếu thủ công ít nhất tiêu đề, số hiệu, ngày ban hành, các mục/nghị quyết, số liệu và phần kết.

### ID `68560`

Metadata cho biết đây là `16/2002/NQ-HĐ`, nhóm `VBQPPL`, hình thức `NQ`, về quỹ quốc phòng - an ninh trên địa bàn tỉnh. OCR text chỉ có 2.056 ký tự. Text chứa tiêu đề, căn cứ, phần `QUYẾT NGHỊ`, một số mức tiền và phần triển khai, nhưng phần cuối bị cắt giữa câu (`H6 t...` trong preview), và nhiều lỗi OCR như `HOLDONG`, `Duk Lak`, `neav`, `nent`.

Độ dài thấp hơn đáng kể so với `133328`, không đủ chứng minh văn bản đã được thu hồi đầy đủ. Status hiện tại là `HTML_VALID` vì fallback đạt ngưỡng 120 ký tự, nhưng đây chỉ là pass kỹ thuật về độ dài, không phải pass về completeness.

Kết luận: không corpus-ready. Nên giữ `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED` và thêm tín hiệu review về khả năng thiếu nội dung; ưu tiên re-OCR hoặc kiểm tra số trang trước mọi bước chunking/indexing.

## Vấn đề logic phát hiện

Ngưỡng `MIN_PDF_TEXT_CHARS = 120` hiện chỉ kiểm tra độ dài text. Nó cho phép text có nhiều lỗi mã hóa hoặc text bị cắt vẫn được nâng status thành `HTML_VALID`. `extraction_confidence` hiện được gán chủ yếu theo phương thức và độ dài: text layer đủ dài nhận `0.95`, OCR đủ dài nhận `0.70`; chưa có điểm chất lượng dựa trên tỷ lệ ký tự lỗi, cấu trúc pháp lý, hoặc completeness.

Do đó, quality gate `content_or_pdf_ratio_gte_95pct` trong pilot đang đo khả năng có content hoặc có PDF, không đo đủ khả năng đưa content vào corpus. Report `READY_FOR_LARGER_PILOT` chỉ nên hiểu là sẵn sàng cho pilot lớn hơn ở tầng thu thập/kỹ thuật, chưa phải sẵn sàng cho corpus RAG hoặc training.

## Cập nhật sau khi thêm quality gate

Đã thêm quality gate ở adapter cho recovered text:

- `RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED`: text layer/OCR chứa `(cid:...)`, ký tự null, hoặc replacement char.
- `OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED`: OCR text dưới 3.000 ký tự nhưng vẫn đủ ngưỡng fallback kỹ thuật 120 ký tự.

Các warning này chỉ là tín hiệu review kỹ thuật. Adapter vẫn giữ `HTML_VALID` cho text đã recover đủ ngưỡng, nhưng harness/pilot report xem record có warning là cần review; raw text và evidence không bị ghi đè.

## Quyết định Phase 2 sau audit

Chưa chuyển sang Phase 3 taxonomy.

Phase 2 đạt ở mức fallback có thể tải PDF, thử text layer/OCR, lưu evidence và tạo warning quality. Phase 2 chưa đạt ở mức corpus-ready vì cả 3 record recovered đều cần review; riêng `138353` cho thấy text layer dài không đồng nghĩa text sạch.

Pilot 120 cũ chưa được chạy lại sau khi thêm gate. Vì vậy chưa được dùng số liệu report cũ để tuyên bố quality gate mới đã pass trên dữ liệu thật.

Các test regression đã pass trên fixture; cần chạy lại exact-manifest pilot 120 sau khi verifier contract pass để cập nhật report thực tế.

## Siết cần làm trước Phase 3

1. Thêm quality audit metrics cho recovered text: tỷ lệ ký tự nghi ngờ, marker `(cid:...)`, tỷ lệ chữ tiếng Việt, mật độ ký tự không hợp lệ, marker cấu trúc (`Điều`, `Khoản`, `Mục`, `QUYẾT NGHỊ`), và dấu hiệu text bị cắt.
2. Tách trạng thái `HTML_VALID` kỹ thuật khỏi quyết định corpus eligibility. Record OCR hoặc record có extraction-quality warning phải vào review queue dù status fetch là `HTML_VALID`.
3. Thêm warning có controlled vocabulary cho text-layer/OCR quality. Không dùng heuristic để kết luận pháp lý; chỉ dùng làm tín hiệu review.
4. Re-OCR thử nghiệm riêng cho `138353` để so sánh text layer lỗi với OCR; không xóa raw text layer cũ.
5. Re-OCR hoặc kiểm tra completeness của `68560`; không cho chunk/index nếu chưa chứng minh đủ nội dung.
6. Thêm regression tests cho trường hợp text đủ 120 ký tự nhưng chất lượng kém hoặc có dấu hiệu truncated.
7. Chạy lại report pilot với hai metric tách biệt: `retrieval_available` và `corpus_ready`.

## Siết cần làm trước Phase 3

1. Thêm quality audit metrics cho recovered text: tỷ lệ ký tự nghi ngờ, marker `(cid:...)`, tỷ lệ chữ tiếng Việt, mật độ ký tự không hợp lệ, marker cấu trúc (`Điều`, `Khoản`, `Mục`, `QUYẾT NGHỊ`), và dấu hiệu text bị cắt.
2. Tách trạng thái `HTML_VALID` kỹ thuật khỏi quyết định corpus eligibility. Record OCR hoặc record có extraction-quality warning phải vào review queue dù status fetch là `HTML_VALID`.
3. Thêm warning có controlled vocabulary cho text-layer/OCR quality. Không dùng heuristic để kết luận pháp lý; chỉ dùng làm tín hiệu review.
4. Re-OCR thử nghiệm riêng cho `138353` để so sánh text layer lỗi với OCR; không xóa raw text layer cũ.
5. Re-OCR hoặc kiểm tra completeness của `68560`; không cho chunk/index nếu chưa chứng minh đủ nội dung.
6. Thêm regression tests cho trường hợp text đủ 120 ký tự nhưng chất lượng kém hoặc có dấu hiệu truncated.
7. Chạy lại report pilot với hai metric tách biệt: `retrieval_available` và `corpus_ready`.

## Kết luận ngắn

Fallback Phase 2 hoạt động đúng về đường đi và evidence preservation: 3 record được thu hồi, 2 bằng OCR, không có lỗi tải PDF. Nhưng audit chất lượng cho thấy không record nào nên được tự động đưa thẳng vào corpus/RAG. `133328` là ứng viên review tốt nhất; `68560` cần re-OCR/completeness check; `138353` cần xử lý chất lượng text layer hoặc OCR đối chứng. Chưa nên qua taxonomy trước khi thêm quality gate cho recovered text.
