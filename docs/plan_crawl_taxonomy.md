# Plan thực thi: crawl VBPL + phân ngành đúng

## Mục tiêu cuối

Mục tiêu cuối của VietLawBERT là cào được văn bản pháp luật Việt Nam từ VBPL, giữ provenance đầy đủ, rồi phân chia chúng theo phân ngành rõ ràng, có thể kiểm chứng, có review queue cho case mơ hồ, và có quality gate đủ chặt để không đưa dữ liệu bẩn vào corpus.

Plan này ưu tiên crawler + taxonomy. Nó không giả định taxonomy có thể suy ra từ URL hoặc free text. Phân ngành phải bám metadata đáng tin, mapping versioned, và review khi thiếu căn cứ.

## Trạng thái hiện tại đã xác minh

Những phần đã sẵn sàng:

- Discovery qua sitemap VBPL đã verify.
- ID đã được giữ dạng string, hỗ trợ numeric, UUID, legacy-prefixed.
- `central`/`local` đã được giữ riêng.
- `official_group` và `official_form` đã được tách đúng.
- Content adapter có trạng thái rõ: `HTML_VALID`, `CONTENT_TOO_SHORT`, `PDF_ONLY`, `METADATA_ONLY`, `SECURITY_CHALLENGE`, `HTTP_ERROR`, `CONTRACT_ERROR`, `PARSE_ERROR`.
- Harness v1 đã có validation, JSONL/GZIP support, duplicate checks, metadata checks, content checks, và real pilot rerun 120 record.
- Pilot 120 thật đã chạy xong bằng exact manifest, có `documents.jsonl.gz`, `checkpoint.jsonl`, `report.json`, `REPORT.md`, `harness_validation_report.json`, `audit_sample.jsonl`, `RUN_INFO.md`, và checksums.

Những gap còn thiếu:

- PDF extraction/OCR fallback production-grade chưa xong.
- Taxonomy phân ngành chuẩn hóa chưa có.
- Review queue taxonomy chưa có.
- Quality gate riêng cho taxonomy coverage/precision chưa có.
- Chưa có quy trình vận hành lớn hơn pilot 120 một cách an toàn.

## Nguyên tắc thiết kế

- Không đoán phân ngành từ URL hoặc free text.
- Không gộp `official_group` với `official_form` thành phân ngành.
- Mọi assignment taxonomy phải truy vết được nguồn.
- Case thiếu/không chắc chắn phải vào review queue, không auto-assign bừa.
- HTTP 200 không đủ để xem là văn bản hợp lệ.
- PDF attachment không đồng nghĩa đã extract text/OCR thành công.
- Ưu tiên correctness và data quality hơn corpus size.

## Phase 1 — Khóa raw schema và contract

### Mục tiêu

Khóa lớp dữ liệu nền để mọi bước sau bám cùng schema.

### Việc cần làm

- Chốt canonical raw record schema từ pilot thật.
- Giữ các field lõi: `document_id`, `source_url`, `status`, `metadata`, `official_group_*`, `official_form_*`, `files`, `warnings`, `scope`, `id_type`, `crawl_timestamp`.
- Duy trì adapter native VBPL → canonical harness record.
- Ghi rõ warning codes và ý nghĩa nghiệp vụ.
- Đồng bộ `docs/data_contract.md`, `harness/adapters/vbpl_record_adapter.py`, và `harness/checks`.

### Deliverable

- Data contract đã chốt.
- Adapter mapping đã chốt.
- Danh sách status/warning codes có nghĩa rõ.

### Quality gate

- 100% native record đi qua adapter mà không fabricate field.
- `HTML_VALID` mới được vào corpus pass trực tiếp.
- Record thiếu provenance bị reject hoặc review rõ ràng.

### Risk

- Contract VBPL đổi khi frontend deploy lại.
- Schema drift giữa artifact thật và canonical record.

### Dependency

- Dựa trên pilot rerun 120 và harness v1 hiện có.

## Phase 2 — PDF extraction / OCR fallback

### Mục tiêu

Đóng gap lớn nhất còn lại: record `PDF_ONLY` và các record `CONTENT_TOO_SHORT` có thể cứu bằng fallback.

### Việc cần làm

- Tách nhánh xử lý PDF text layer và scanned PDF.
- Lưu extraction outcome riêng: method, source PDF metadata, confidence/basic flags.
- Không xem có file PDF là corpus pass.
- Gắn record không extract được hoặc text quá ngắn vào review queue.
- Thêm test fixture cho PDF text, PDF scan, PDF lỗi, PDF corrupted.

### Deliverable

- Module PDF fallback.
- OCR fallback tối thiểu.
- Báo cáo coverage trên subset fail của pilot 120.

### Quality gate

- Record recoverable từ PDF phải tăng lên có kiểm chứng.
- PDF presence không được tự động biến thành pass.

### Risk

- OCR tiếng Việt sai dấu.
- Layout pháp lý bị vỡ.
- Chi phí runtime tăng.

### Dependency

- Sau Phase 1.

## Phase 3 — Thiết kế taxonomy phân ngành

### Mục tiêu

Tạo taxonomy phân ngành rõ ràng, versioned, và có thể kiểm chứng nguồn.

### Việc cần làm

- Định nghĩa taxonomy nhiều tầng, tối thiểu gồm domain lớn và phân ngành con.
- Chỉ dùng nguồn đáng tin để gán taxonomy: metadata chính thức, scope, official group/form, và bảng mapping có kiểm duyệt.
- Không dùng URL slug, title, body text làm nguồn quyết định cuối.
- Tách riêng taxonomy ngành với `official_group` và `official_form`.
- Định nghĩa trạng thái taxonomy: `AUTO_ASSIGNED_TRUSTED`, `REVIEW_REQUIRED`, `UNCLASSIFIED`.

### Deliverable

- Taxonomy spec v1.
- Mapping table raw metadata → taxonomy node.
- Review guideline cho case conflict/missing.

### Quality gate

- Mỗi assignment có evidence truy vết được.
- 0 rule auto-assign cuối dựa trên URL/free text.

### Risk

- Metadata lĩnh vực VBPL có thể thiếu hoặc không nhất quán.
- Taxonomy quá chi tiết sớm gây backlog review lớn.

### Dependency

- Cần khảo sát metadata thật từ pilot và sample mở rộng.

## Phase 4 — Nối taxonomy vào pipeline và harness

### Mục tiêu

Biến taxonomy thành bước kỹ thuật thật trong pipeline, không phải thao tác tay.

### Việc cần làm

- Thêm taxonomy normalization layer sau adapter.
- Record đủ tin cậy thì auto-map.
- Record thiếu hoặc conflict thì vào review queue.
- Mở rộng harness để validate taxonomy node, source evidence, conflict rule, và version.
- Tách output thành corpus-ready, taxonomy review queue, và fallback queue.

### Deliverable

- Taxonomy-normalized record shape.
- Harness checks cho taxonomy.
- Review queue format ổn định.

### Quality gate

- Record vào corpus phải có content pass + taxonomy state hợp lệ.
- Không có auto-assign mơ hồ mà không có evidence.

### Risk

- Nếu biến taxonomy thành hard gate quá sớm, throughput giảm mạnh.

### Dependency

- Phase 2 và Phase 3.

## Phase 5 — Pilot trung gian 500–1.000 record

### Mục tiêu

Xác minh hệ thống ở quy mô trung gian trước khi nghĩ đến full sitemap.

### Việc cần làm

- Mở rộng pilot theo strata hiện có: `scope × id_type × observed_group`.
- Bổ sung sampling theo official group/form nếu cần.
- Chạy cả HTML valid, PDF fallback, metadata-only.
- Audit thủ công một sample taxonomy đủ đại diện.
- Đo riêng content quality, fallback recovery, taxonomy coverage, taxonomy review ratio, and precision trên sample review.

### Deliverable

- Pilot report 500–1.000 record.
- Báo cáo taxonomy quality.
- Báo cáo fallback recovery.

### Quality gate

- Crawler gate pass.
- Taxonomy trusted auto-assign precision đạt ngưỡng.
- Review ratio có kiểm soát.

### Risk

- Sample review không đại diện.
- Backlog review tăng nhanh.

### Dependency

- Phase 2 và 4.

## Phase 6 — Crawl production theo queue phân lớp

### Mục tiêu

Chạy lớn nhưng kiểm soát được chất lượng và backlog review.

### Việc cần làm

- Chia queue rõ: HTML-valid, PDF/OCR, metadata-only, taxonomy review.
- Chạy checkpoint/resume.
- Verifier định kỳ trước batch.
- Theo dõi drift: status ratio, fallback ratio, missing metadata, taxonomy coverage, review backlog.

### Deliverable

- Batch reports định kỳ.
- Review backlog inventory.
- Dữ liệu corpus đã phân ngành rõ.

### Quality gate

- Dừng batch nếu tăng đột biến `SECURITY_CHALLENGE`, `CONTRACT_ERROR`, `PARSE_ERROR`.
- Dừng nếu taxonomy review ratio vượt ngưỡng.

### Risk

- Drift frontend.
- Rate limit.
- Review backlog phình quá nhanh.

### Dependency

- Chỉ nên làm sau khi pilot trung gian pass.

## Thứ tự triển khai đề xuất

1. Khóa schema và contract.
2. Làm PDF/OCR fallback.
3. Thiết kế taxonomy spec + mapping table.
4. Nối taxonomy vào harness và review queue.
5. Chạy pilot 500–1.000 với gate kép.
6. Mới tính full crawl.

## Các file ưu tiên nếu bắt đầu implement ngay

- `law_dataset/src/crawler/vbpl_content_adapter.py`
- `law_dataset/src/crawler/pilot_crawl_vbpl.py`
- `harness/adapters/vbpl_record_adapter.py`
- `harness/checks/validate_metadata.py`
- `harness/checks/validate_pipeline.py`
- `docs/data_contract.md`
- `docs/crawling_rules.md`

## Ghi chú thực tế

Phase 2 và 3 là hai điểm nghẽn thật. Nếu chưa có PDF/OCR fallback và taxonomy spec, không nên scale lớn. Pilot 120 đã đủ chứng minh rằng đường đi đúng hiện tại là: tăng cường fallback, rồi mới mở rộng taxonomy và pilot.
