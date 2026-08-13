# VietLawBERT / VIETLAWCRAWL

Repository này phục vụ xây dựng corpus và pipeline AI cho văn bản pháp luật Việt Nam. Hướng hiện tại: khám phá URL qua VBPL sitemap, lấy metadata công khai, dùng content adapter riêng cho HTML/PDF, validate chất lượng, rồi mới làm sạch, chunk, embedding, retrieval, và đánh giá RAG/model.

## Harness Engineering v1

Harness v1 thêm lớp kiểm tra có thể chạy tự động để tránh đưa dữ liệu lỗi vào corpus. Nó kiểm tra nội dung crawl, metadata, duplicate URL/document ID/content hash, JSON/JSONL, và tạo report JSON nhỏ tại `harness/reports/latest_validation.json`. Với record native VBPL, cần đi qua adapter `harness.adapters.vbpl_record_adapter` trước khi validate. Adapter canonical hóa thêm `content_chars`, `content_sha256`, `warning_codes`, `scope`, `id_type`, `observed_group`, `backend`, và `error` nhưng không fabricate field thiếu. Với contract hiện tại, chỉ record `HTML_VALID` mới được xem là corpus record hợp lệ; `CONTENT_TOO_SHORT`, `PDF_ONLY`, và `METADATA_ONLY` là trạng thái review/fallback, chưa phải pass corpus. Warning codes được xem như vocabulary có kiểm soát, không phải chuỗi tự do tùy ý.

Chạy validation mặc định trên fixture synthetic:

```bash
python -m harness.checks.validate_pipeline
```

Chạy trên file cụ thể:

```bash
python -m harness.checks.validate_pipeline --input path/to/records.jsonl --report harness/reports/latest_validation.json
```

Ví dụ PowerShell:

```powershell
python -m harness.checks.validate_pipeline --input .\harness\fixtures\sample_documents.jsonl
```

## Kiểm tra sau thay đổi crawler

```bash
python -m py_compile law_dataset/src/crawler/*.py
python -m unittest discover -s law_dataset/tests -v
python -m unittest discover -s tests -v
python -m harness.checks.validate_pipeline --input harness/fixtures/sample_documents.jsonl
git diff --check
```

## Tài liệu quan trọng

Đọc `AGENTS.md`, `docs/architecture.md`, `docs/data_contract.md`, `docs/crawling_rules.md`, `law_dataset/VBPL_CONTENT_ADAPTER.md`, `law_dataset/VBPL_VERIFICATION_FINDINGS.md`, và `law_dataset/artifacts/vbpl_pilot_120/REPORT.md` trước khi thay đổi pipeline crawl.
