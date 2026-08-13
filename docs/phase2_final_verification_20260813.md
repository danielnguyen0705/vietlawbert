# Phase 2 final verification — 2026-08-13

## Phạm vi

Xác minh cuối cho Phase 2 sau khi thêm recovered-text remediation, rerun 3 record blocked (`138353`, `133328`, `68560`), và rebuild pilot report từ evidence mới.

## Kết quả kiểm tra bắt buộc

- `python -m py_compile law_dataset/src/crawler/*.py` — pass.
- `python -m unittest discover -s law_dataset/tests -v` — pass.
- `python -m unittest discover -s tests -v` — pass.
- `git diff --check` — fail do trailing whitespace ở file ngoài phạm vi thay đổi hiện tại.

Các file ngoài phạm vi hiện làm `git diff --check` fail gồm ít nhất:

- `requirements.txt`
- `law_dataset/tests/test_verify_vbpl_contract.py`

Theo rule bảo toàn thay đổi không liên quan, các file này chưa được sửa trong vòng này.

## Kết quả pilot sau rerun

Nguồn report cuối:

- `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/report.json`
- `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/REPORT.md`

Kết quả chính:

- Technical retrieval decision: `READY_FOR_LARGER_PILOT`
- Corpus readiness decision: `REVIEW_REQUIRED`
- `corpus_ready_records = 114`
- `corpus_review_required_records = 3`
- `recovered_quality_review_records = 3`

## Trạng thái 3 record blocked

### `138353`

- `HTML_VALID`
- `PDF_TEXT_LAYER`
- Bị block corpus bởi:
  - `RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED`
  - `RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED`
  - `RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED`
- Metrics chính:
  - `suspect_token_count = 1899`
  - `suspect_char_density = 0.019311144330211416`
  - `vietnamese_char_ratio = 0.0005700673901093307`
  - `structure_marker_count = 224`

Đánh giá: text layer dài nhưng nhiễu nặng, chưa corpus-ready.

### `133328`

- `HTML_VALID`
- `PDF_OCR`
- Bị block corpus bởi:
  - `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED`
- Metrics chính:
  - `suspect_token_count = 0`
  - `suspect_char_density = 0.0`
  - `vietnamese_char_ratio = 0.07050686487044493`
  - `structure_marker_count = 21`

Đánh giá: record recovered tốt nhất trong 3 record; còn cần OCR review trước khi vào corpus.

### `68560`

- `HTML_VALID`
- `PDF_OCR`
- Bị block corpus bởi:
  - `CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED`
  - `OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED`
  - `RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED`
  - `RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED`
- Metrics chính:
  - `suspect_token_count = 1`
  - `suspect_char_density = 0.00048638132295719845`
  - `vietnamese_char_ratio = 0.04926470588235294`
  - `structure_marker_count = 3`

Đánh giá: OCR ngắn, còn tín hiệu quality risk, chưa corpus-ready.

## Kết luận rollout boundary

Phase 2 technical retrieval đã ổn ở mức pilot 120. Phase 2 corpus readiness chưa đạt. Chưa sang Phase 3 taxonomy. Chưa chạy pilot 500–1.000. Chưa chạy full sitemap crawl.

## Bước tiếp theo đề nghị

1. Audit tay `133328` trước, vì đây là record recovered sạch nhất.
2. Re-extract hoặc OCR đối chứng cho `138353`.
3. Re-OCR/completeness check cho `68560`.
4. Chỉ cân nhắc mở corpus gate khi `corpus_review_required_records` về `0`.
