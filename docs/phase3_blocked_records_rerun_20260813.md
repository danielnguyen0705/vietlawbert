# Phase 3 blocked records rerun — 2026-08-13

## Exact manifest assertions

- Manifest được tạo bằng parse JSONL từ artifact canonical, không copy dòng thủ công.
- Số record: `3`.
- `document_id`: `133328`, `138353`, `68560`.
- `source_url` của cả 3 record đều không rỗng và unique.
- `metadata` trong manifest được copy nguyên từ artifact canonical `vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/checkpoint.jsonl`.

## Manual review raw-evidence rule

- Native output giữ raw evidence nếu `manual_corpus_review` sai kiểu qua field `manual_corpus_review_raw`.
- Normalized harness record vẫn có thể drop field invalid.
- Không auto-approve record nào.

## pdf_page_count policy

- Chưa có producer + provenance thật trong adapter/output hiện tại.
- Vì vậy `pdf_page_count` vắng mặt ở toàn bộ output rerun này.

## Per-record evidence

### `133328`

- `status`: `HTML_VALID`
- `extraction_method`: `PDF_OCR`
- `recovered_text_selection`: `{"selected_method": "PDF_OCR", "reason": "text_layer_too_short_ocr_selected", "text_layer_chars": 0, "ocr_chars": 13716}`
- `pdf_text_chars`: `0`
- `pdf_ocr_chars`: `13716`
- `content_chars`: `13716`
- `recovered_text_metrics`: `{"total_chars": 13716, "suspect_token_count": 0, "suspect_char_density": 0.0, "vietnamese_char_ratio": 0.07050686487044493, "structure_marker_count": 21, "trailing_fragment_suspected": false}`
- `warning_after`: `["CONTENT_RECOVERED_FROM_PDF", "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"]`
- `corpus_decision`: `REVIEW_REQUIRED`
- `corpus_blockers`: `["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"]`
- `manual_corpus_review`: `null`
- `manual_corpus_review_raw`: `null`
- `pdf_page_count`: `null`

- Manual review vẫn pending ở mức nghiệp vụ. Output không có approval field.
- Evidence pack cho người thật: dùng `checkpoint.jsonl`, `documents.jsonl.gz`, `report.json`, và attachment PDF qua `pdf_url` trong record.

### `138353`

- `status`: `HTML_VALID`
- `extraction_method`: `PDF_TEXT_LAYER`
- `recovered_text_selection`: `{"text_layer_method": "PDF_TEXT_LAYER", "text_layer_chars": 98337, "text_layer_metrics": {"total_chars": 98337, "suspect_token_count": 1899, "suspect_char_density": 0.019311144330211416, "vietnamese_char_ratio": 0.0005700673901093307, "structure_marker_count": 224, "trailing_fragment_suspected": false}, "text_layer_warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED", "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED"], "text_layer_score": -373238.4137591201, "ocr_attempted_for_selection": true, "selected_method": "PDF_TEXT_LAYER", "reason": "text_layer_kept_after_ocr_comparison", "ocr_method": "OCR_DISABLED", "ocr_chars": 0, "ocr_metrics": {"total_chars": 0, "suspect_token_count": 0, "suspect_char_density": 0.0, "vietnamese_char_ratio": 0.0, "structure_marker_count": 0, "trailing_fragment_suspected": false}, "ocr_warnings": [], "ocr_score": -500.0}`
- `pdf_text_chars`: `98337`
- `pdf_ocr_chars`: `0`
- `content_chars`: `98337`
- `recovered_text_metrics`: `{"total_chars": 98337, "suspect_token_count": 1899, "suspect_char_density": 0.019311144330211416, "vietnamese_char_ratio": 0.0005700673901093307, "structure_marker_count": 224, "trailing_fragment_suspected": false}`
- `warning_after`: `["CONTENT_RECOVERED_FROM_PDF", "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED", "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED"]`
- `corpus_decision`: `REVIEW_REQUIRED`
- `corpus_blockers`: `["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED", "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED"]`
- `manual_corpus_review`: `null`
- `manual_corpus_review_raw`: `null`
- `pdf_page_count`: `null`

- OCR không được promote. Selection giữ `PDF_TEXT_LAYER` vì output adapter hiện hành không chứng minh OCR tốt hơn candidate đang chọn.
- Record vẫn fail corpus do metrics text layer rất xấu.

### `68560`

- `status`: `HTML_VALID`
- `extraction_method`: `PDF_OCR`
- `recovered_text_selection`: `{"selected_method": "PDF_OCR", "reason": "text_layer_too_short_ocr_selected", "text_layer_chars": 0, "ocr_chars": 2056}`
- `pdf_text_chars`: `0`
- `pdf_ocr_chars`: `2056`
- `content_chars`: `2056`
- `recovered_text_metrics`: `{"total_chars": 2056, "suspect_token_count": 1, "suspect_char_density": 0.00048638132295719845, "vietnamese_char_ratio": 0.04926470588235294, "structure_marker_count": 3, "trailing_fragment_suspected": false}`
- `warning_after`: `["CONTENT_RECOVERED_FROM_PDF", "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED", "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED"]`
- `corpus_decision`: `REVIEW_REQUIRED`
- `corpus_blockers`: `["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED", "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED"]`
- `manual_corpus_review`: `null`
- `manual_corpus_review_raw`: `null`
- `pdf_page_count`: `null`

- Không kết luận thiếu trang. `pdf_page_count` không có producer/provenance thật nên để vắng mặt.

## Output files

- `law_dataset/artifacts/vbpl_phase3_blocked_rerun_20260813/sample_manifest.jsonl`
- `law_dataset/artifacts/vbpl_phase3_blocked_rerun_20260813/checkpoint.jsonl`
- `law_dataset/artifacts/vbpl_phase3_blocked_rerun_20260813/documents.jsonl.gz`
- `law_dataset/artifacts/vbpl_phase3_blocked_rerun_20260813/report.json`
- `law_dataset/artifacts/vbpl_phase3_blocked_rerun_20260813/REPORT.md`
