# Phase 3 repository inventory — 2026-08-13

## 1. Git trạng thái đầu vào

- Branch làm việc: `cuong-ubutu`
- HEAD commit: `094693691b665f59c83773e1764de082400b20db`
- HEAD short: `0946936 (HEAD -> cuong-ubutu, origin/cuong-ubutu) up and test`
- Remote push/fetch: `https://github.com/danielnguyen0705/vietlawbert.git`
- Python: `3.10.12`

## 2. Working tree trước khi agent tiếp tục Milestone 1

Working tree đã bẩn sẵn, không được dọn bằng lệnh phá huỷ. Tại thời điểm kiểm kê có `66` file modified. Các file tiêu biểu ngoài phạm vi Milestone 1 gồm:

- `.dockerignore`
- `.gitignore`
- `CLAUDE.md`
- `Dockerfile`
- `app.py`
- `benchmark/evaluate_rrf.py`
- `benchmark/multi_hop.jsonl`
- `benchmark/single_hop.jsonl`
- `docker-compose.yml`
- `harness/reports/latest_validation.json`
- `law_dataset/.env.example`
- `law_dataset/README.md`
- `law_dataset/VBPL_CONTENT_ADAPTER.md`
- `law_dataset/VBPL_PROBE.md`
- `law_dataset/VBPL_VERIFICATION_FINDINGS.md`
- nhiều artifact/report cũ dưới `law_dataset/artifacts/`
- nhiều module crawler/database/training hiện hữu dưới `law_dataset/src/`
- `requirements.txt`

Kết luận Git-safe:

- Repo không ở trạng thái sạch.
- Không được dùng `git add .`, `git reset --hard`, `git clean -fd`, `git checkout -- <file>`.
- Mọi file tạo/sửa mới cho Phase 3 phải stage chọn lọc riêng.

## 3. File/chỉ dẫn nền đã kiểm tra

Đã đối chiếu các nguồn nền bắt buộc trước khi tiếp tục:

- `CLAUDE.md`
- `AGENTS.md`
- `README.md`
- `docs/data_contract.md`
- `docs/phase2_final_verification_20260813.md`
- `law_dataset/VBPL_CONTENT_ADAPTER.md`
- `law_dataset/VBPL_VERIFICATION_FINDINGS.md`
- `law_dataset/artifacts/vbpl_pilot_120/REPORT.md`
- `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/REPORT.md`
- `HUONG_DAN_AI_AGENT_VBPL_PHASE3.md`

## 4. Entry points thật cho crawl / rerun / report

Qua kiểm tra CLI trong `law_dataset/src/crawler/`:

### 4.1. Pilot/rerun chính

File: `law_dataset/src/crawler/pilot_crawl_vbpl.py`

Có các entrypoint thật:

- `def build_parser()` tại dòng ~491
- `def main()` tại dòng ~520
- `if __name__ == "__main__":` tại dòng ~596

Vai trò hiện tại:

- đọc `--input` hoặc `--manifest`
- hỗ trợ exact-manifest replay
- ghi `sample_manifest.jsonl`
- ghi `checkpoint.jsonl`
- ghi `documents.jsonl.gz`
- build `report.json` và `REPORT.md`
- dừng sớm khi gặp `SECURITY_CHALLENGE`, `CONTRACT_ERROR`, `PARSE_ERROR`

### 4.2. Probe/khảo sát hợp đồng

File: `law_dataset/src/crawler/probe_vbpl.py`

Có các entrypoint thật:

- `def build_parser()` tại dòng ~417
- `def main()` tại dòng ~435
- `if __name__ == "__main__":` tại dòng ~541

Vai trò: probe/smoke/contract-level verification, không phải pipeline pilot canonical chính.

### 4.3. Harness validation

File: `harness/checks/validate_pipeline.py`

Vai trò:

- nhận input JSON/JSONL/JSONL.GZ
- auto-adapt native VBPL record qua `harness.adapters.vbpl_record_adapter`
- chạy `validate_document_content`, `validate_metadata`, duplicate checks
- xuất report JSON nhỏ

## 5. Artifact hiện có và mức độ sẵn dùng

### 5.1. Artifact pilot/report đã xác nhận tồn tại

Các thư mục artifact quan trọng đã hiện diện:

- `law_dataset/artifacts/vbpl_pilot_120/`
  - `REPORT.md`
  - `report.json`
  - `sample_manifest.jsonl`
- `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_20260813/`
  - `REPORT.md`
  - `report.json`
  - `sample_manifest.jsonl`
- `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/`
  - `REPORT.md`
  - `report.json`
  - `sample_manifest.jsonl`
  - `checkpoint.jsonl`
  - `documents.jsonl.gz`
- `law_dataset/artifacts/vbpl_pilot_120_phase2_fallback_20260813/`
- `law_dataset/artifacts/vbpl_pilot_120_phase2_quality_gates_20260813/`
- `law_dataset/artifacts/vbpl_content_audit_120/`
- các artifact probe/contract verification liên quan

### 5.2. Artifact Milestone 1 còn thiếu

Chưa thấy artifact riêng cho Phase 3 blocked rerun:

- `law_dataset/artifacts/vbpl_phase3_blocked_rerun_20260813/`

Chưa thấy report inventory cho Phase 3 trước khi tạo file này.

## 6. Baseline Phase 2 thật đang nói gì

Từ `law_dataset/artifacts/vbpl_pilot_120_phase2_corpus_gates_rerun_20260813/REPORT.md`:

- Technical retrieval decision: `READY_FOR_LARGER_PILOT`
- Corpus readiness decision: `REVIEW_REQUIRED`
- `records = 120`
- `corpus_ready_records = 114`
- `corpus_review_required_records = 3`
- `recovered_quality_review_records = 3`
- `recovered_from_pdf = 3`
- `recovered_by_ocr = 2`
- `pdf_fetch_failed = 0`
- `pdf_extraction_insufficient = 2`

Ba record blocked còn lại theo artifact/rerun/report hiện hữu:

- `133328`
- `138353`
- `68560`

## 7. Baseline tests trước khi sửa tiếp

Đã chạy lại các lệnh baseline bắt buộc trong repo hiện tại:

### 7.1. `python -m py_compile law_dataset/src/crawler/*.py`

- Kết quả: pass
- Thời gian: `real 0.91s`

### 7.2. `python -m unittest discover -s law_dataset/tests -v`

- Kết quả: pass
- Số test: `81`
- Thời gian: `real 0.86s`

### 7.3. `python -m unittest discover -s tests -v`

- Kết quả: pass
- Số test: `109`
- Thời gian: `real 0.91s`

### 7.4. `python -m harness.checks.validate_pipeline --input harness/fixtures/sample_documents.jsonl`

- Kết quả: `PASS`
- Documents scanned: `2`
- Rejected: `0`
- Duplicate documents: `0`
- Thời gian: `real 0.41s`

### 7.5. `git diff --check`

- Kết quả: fail
- Bản chất: baseline repository issue ngoài phạm vi Milestone 1, không phải regression mới của thay đổi Phase 3 đang chuẩn bị.
- Dấu hiệu: trailing whitespace xuất hiện hàng loạt ở file có sẵn trong working tree như `.dockerignore`, `Dockerfile`, `app.py`, `law_dataset/README.md`, `law_dataset/VBPL_CONTENT_ADAPTER.md`, `law_dataset/VBPL_VERIFICATION_FINDINGS.md`, artifact markdown/json cũ, `requirements.txt`, v.v.
- Theo guide Phase 3 và project rules: không được sửa các file ngoài phạm vi chỉ để làm đẹp `git diff --check`.

## 8. Khác biệt giữa working tree thực tế và mô tả ở Mục 2 của guide

### 8.1. Khớp với mô tả

Các điểm sau đã hiện diện thật trong code/harness:

- `vbpl_content_adapter.py` có PDF fallback, OCR, recovered-text metrics/warnings
- có `recovered_text_selection` trong `AdapterResult`
- có `manual_corpus_review` trong `AdapterResult`
- `harness/adapters/vbpl_record_adapter.py` preserve `recovered_text_selection` và `manual_corpus_review`
- `harness/checks/validate_metadata.py` enforce manual override có kiểm soát
- test cho OCR selection/manual review đã tồn tại và đang pass

### 8.2. Chưa khớp hoàn toàn / còn thiếu so với yêu cầu Milestone 1+

- `pilot_crawl_vbpl.py` hiện chưa ghi `recovered_text_selection` vào `make_record(...)` dù `AdapterResult` có field này.
- `pilot_crawl_vbpl.py` hiện chưa ghi `manual_corpus_review` vào record output.
- Chưa thấy field `pdf_page_count` hoặc page-count evidence trong record/report của pilot hiện tại, trong khi guide Phase 3 yêu cầu so sánh page count cho blocked rerun.
- Chưa có artifact Phase 3 riêng cho rerun ba blocked record.
- Chưa có `docs/phase3_blocked_records_rerun_20260813.md`.
- Chưa có benchmark manifest 30 record của Phase 3.
- Chưa có benchmark candidate Crawlee/Scrapling theo vai trò tách biệt.

## 9. Đánh giá readiness để tiếp tục Milestone 1

Đã hoàn tất phần đầu của Milestone 1:

- kiểm tra branch/Git trạng thái đầu vào
- đọc instruction + repo structure
- chạy baseline tests
- xác minh artifact pilot thật
- lập inventory có file hoá

Chưa hoàn tất Milestone 1:

- rerun đúng ba blocked record vào artifact Phase 3 mới
- tạo report so sánh trước/sau cho `133328`, `138353`, `68560`
- tách quyết định baseline Phase 3 thành `technical_retrieval_decision`, `corpus_readiness_decision`, `corpus_ready_count`, `review_required_count`, `retrieval_failed_count`

## 10. Next step trực tiếp

Bước tiếp theo đúng thứ tự guide:

1. tạo exact manifest chỉ gồm `133328`, `138353`, `68560` từ artifact pilot hiện hữu;
2. rerun bằng adapter hiện tại vào thư mục mới `law_dataset/artifacts/vbpl_phase3_blocked_rerun_20260813/`;
3. rebuild `report.json` và `REPORT.md` mới;
4. viết `docs/phase3_blocked_records_rerun_20260813.md` với so sánh method, content length, metrics, warnings, corpus decision, và evidence pack cho `133328`.
