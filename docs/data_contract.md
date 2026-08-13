# Data contract cho VietLawBERT corpus

Schema dưới đây mô tả contract chuẩn hóa cho harness. Nó bám các record hiện có trong `law_dataset/src/crawler/pilot_crawl_vbpl.py`, `law_dataset/artifacts/vbpl_pilot_120/report.json`, và các test crawler. Không ép nguồn phải cung cấp trường chưa đáng tin cậy.

| Field | Nhóm | Type | Nullable | Ý nghĩa | Ví dụ | Validation | Normalization |
|---|---|---|---:|---|---|---|---|
| `document_id` | required | string | false | ID ổn định từ VBPL, hỗ trợ numeric, UUID, legacy-prefixed | `175440`, `vbpqta_6368` | không rỗng, giữ dạng string | trim, không cast sang int |
| `source_url` | required khi có page source | string | false | URL nguồn công khai | `https://vbpl.vn/van-ban/chi-tiet/175440` | URL không rỗng, dùng để dedupe | lower scheme/host, bỏ fragment/query khi dedupe |
| `status` | required | string enum | false | kết quả content adapter | `HTML_VALID` | thuộc nhóm trạng thái đã định nghĩa; chỉ `HTML_VALID` là corpus-pass trực tiếp, các trạng thái fallback/review phải bị chặn khỏi accepted corpus | upper-case |
| `title` | recommended | string | true | tên văn bản nếu nguồn cung cấp | `Nghị định ...` | không fail nếu thiếu, trừ khi config yêu cầu | trim whitespace |
| `official_group_code` | recommended | string | true | nhóm chính thức từ VBPL | `VBQPPL` | thiếu thì warning/review | giữ nguyên mã nguồn |
| `official_form_code` | recommended | string | true | hình thức văn bản từ VBPL | `NĐ`, `QĐ` | không dùng thay thế group | giữ riêng với group |
| `official_form_name` | recommended | string | true | tên hình thức văn bản | `Nghị định` | không suy ra lĩnh vực | trim |
| `document_number` / `docNum` | recommended | string | true | số/ký hiệu văn bản | `01/2026/NĐ-CP` | nếu có thì trim | giữ nguyên dấu, slash |
| `issuer` | optional | string | true | cơ quan ban hành nếu nguồn cung cấp | `Chính phủ` | không chuẩn hóa bằng đoán URL | giữ raw, map bằng bảng riêng nếu có |
| `issued_date` | recommended | string/date | true | ngày ban hành | `2026-01-01` | ISO hoặc `dd/mm/yyyy` | chuyển ISO ở pipeline chuẩn hóa nếu chắc chắn |
| `effective_date` | optional | string/date | true | ngày hiệu lực | `2026-02-01` | format hợp lệ nếu có | như trên |
| `expiration_date` | optional | string/date | true | ngày hết hiệu lực | `null` | format hợp lệ nếu có | như trên |
| `central_local` / `scope` | recommended | string enum | true | phạm vi nguồn sitemap | `central`, `local` | không bịa nếu thiếu | lower-case |
| `field` | optional | string | true | lĩnh vực/chủ đề nếu metadata tin cậy có | `đất đai` | không suy đoán từ URL | bảng chuẩn hóa riêng |
| `content` / `html` | required cho corpus accepted | string | true với fallback | nội dung pháp luật đã trích | `Điều 1...` | không rỗng, đủ dài, không security page | UTF-8, normalize whitespace khi hash |
| `content_sha256` | derived | string | true | hash nội dung hợp lệ | SHA-256 hex | chỉ tính cho full content hợp lệ | SHA-256 của normalized text |
| `files` | optional | array[object] | true | attachment/PDF metadata đã lọc | `[{"fileName":"original.pdf"}]` | chỉ giữ item object; PDF không đồng nghĩa OCR pass | giữ metadata nhỏ |
| `warnings` | derived | array[string] | true | cảnh báo review raw từ crawler/adapter | `MISSING_OFFICIAL_PARENT_GROUP` | không bỏ qua im lặng | mã ổn định, trim whitespace |
| `warning_codes` | derived | array[string] | true | alias canonical của `warnings` cho harness/report | `DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED` | mỗi phần tử phải là string không rỗng | đồng bộ với `warnings` |
| `content_chars` | derived | integer | true | số ký tự của `content` tại thời điểm lưu raw record | `8421` | nếu có phải khớp nội dung hoặc được suy ra lại | số nguyên không âm |
| `content_sha256` | derived | string | true | hash nội dung hợp lệ/raw đã trích | SHA-256 hex | giữ nếu source record đã có; duplicate gate chỉ áp cho full content hợp lệ | SHA-256 của normalized text khi dedupe corpus |
| `recovered_text_normalized` | derived | string | true | bản whitespace-normalized song song của text PDF/OCR recovered | `Điều 1...` | chỉ có khi fallback recover thành công; không thay `content_html` raw | CRLF về LF, space/tab liên tiếp co lại, tối đa hai newline liên tiếp |
| `recovered_text_normalization` | derived | object | true | provenance của normalization recovered text | `{\"method\":\"PDF_OCR\",\"flags\":[\"whitespace_normalized\"]}` | chỉ nhận object; phải giữ method và flags audit được | không suy diễn hoặc fabricate field sai kiểu |
| `recovered_text_metrics` | derived | object | true | metrics review của raw recovered text | `{\"suspect_char_density\":0.02,\"trailing_fragment_suspected\":false}` | chỉ nhận object; heuristic là review signal, không phải kết luận pháp lý | tính từ raw text, không tính từ normalized text |
| `crawl_timestamp` | recommended | string datetime | true | thời điểm crawl | ISO 8601 | datetime hợp lệ nếu có | UTC hoặc timezone rõ |
| `backend` | optional | string | true | backend/path lấy nội dung | `server_action` | không bắt buộc | trim |
| `error` | optional | string | true | lỗi kỹ thuật raw nếu fetch thất bại | `Unexpected content type: text/html` | chỉ là evidence, không biến lỗi thành corpus | trim |
| `observed_group` | recommended | string | true | nhóm quan sát từ public metadata/JSON-LD | `LEGAL_FORM_CANDIDATE` | không đồng nhất với `official_group_code` | giữ nguyên |
| `id_type` | recommended | string enum | true | family của document ID | `numeric`, `uuid`, `legacy_prefixed` | không bịa nếu thiếu | lower-case |

## Warning codes chuẩn hóa hiện tại

Các mã dưới đây đã xuất hiện trong adapter/pilot hiện tại và nên được xem là vocabulary ổn định của Phase 2:

- `MISSING_OFFICIAL_PARENT_GROUP`
- `MISSING_DOCUMENT_NUMBER_FOR_GROUP_LOOKUP`
- `OFFICIAL_GROUP_NOT_FOUND`
- `DUPLICATE_ID_IN_GROUP_LOOKUP`
- `DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED`
- `OFFICIAL_GROUP_ADMINISTRATIVE_FLAG_CONFLICT`
- `CONTENT_RECOVERED_FROM_PDF`
- `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED`
- `RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED`
- `OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED`
- `PDF_EXTRACTION_INSUFFICIENT`
- `PDF_FETCH_FAILED`

Hai warning mới của bước siết chất lượng recovered text có ý nghĩa hẹp:

- `RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED`: text recovered có dấu hiệu lỗi mã hóa/ký tự rác như `(cid:...)`, ký tự null, hoặc replacement char; chỉ là tín hiệu review kỹ thuật.
- `OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED`: OCR đã recover đủ để qua fallback kỹ thuật nhưng vẫn ngắn bất thường, cần review completeness trước khi đưa vào corpus/RAG.

Nếu phát sinh warning mới, phải thêm test và cập nhật contract thay vì để crawler/harness ngầm chấp nhận mọi chuỗi tự do.

## Chunk schema hiện tại/khuyến nghị

Pipeline training có nhắc tới `final_contextual_chunks.jsonl`; harness v1 validate JSONL theo contract tối thiểu:

```json
{
  "document_id": "sample-001",
  "source_url": "https://vbpl.vn/van-ban/chi-tiet/sample-001",
  "field_slug": "synthetic",
  "chunk_index": 0,
  "text": "Điều 1. Phạm vi điều chỉnh..."
}
```

`document_id`, `chunk_index`, và `text` là lõi cho chunk validation. `text` phải là string không rỗng. `chunk_index` phải là số nguyên không âm. Nếu có `source_url`, phải giữ provenance từ văn bản gốc.
