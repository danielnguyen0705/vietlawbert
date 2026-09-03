# Tài liệu VietLawBERT

## Bắt đầu từ đâu?

- Muốn chạy toàn bộ pipeline crawl: đọc [guides/ubuntu_to_full_crawl.md](guides/ubuntu_to_full_crawl.md).
- Muốn chạy pilot nhỏ: đọc [guides/pilot_100_runbook.md](guides/pilot_100_runbook.md).
- Muốn xem kết quả kiểm định 8.000 văn bản: đọc
  [reports/crawl_method_8000_validation.md](reports/crawl_method_8000_validation.md).
- Muốn xem các tối ưu crawler/consumer: đọc
  [reports/phase1_5_stabilization.md](reports/phase1_5_stabilization.md).

## Phân loại

- `guides/`: hướng dẫn vận hành có thể thực thi.
- `reports/`: kết quả đo, gate chất lượng và quyết định kỹ thuật hiện hành.
- `archive/`: handoff, inventory và báo cáo lịch sử; không dùng làm runbook hiện tại.

## Entry point source

Các tác vụ vận hành chạy từ `law_dataset/src` bằng `python -m cli.<tác_vụ>`:

- `cli.crawl`, `cli.ocr`: crawl và OCR.
- `cli.audit`, `cli.verify`: quality gate.
- `cli.publish`, `cli.consume_raw`, `cli.consume_embeddings`: Kafka và ingestion.
- `cli.inspect_database`: kiểm tra nhanh Milvus/Neo4j.
