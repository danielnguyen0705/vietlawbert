# Phương pháp crawl và báo cáo kiểm định 8.000 văn bản pháp luật

*Ngày kiểm định: 16/08/2026*

## 1. Phương pháp

Crawler dùng hướng API-first:

1. Playwright mở một browser context để lấy cookie/action token của VBPL.
2. Search server action trả danh sách theo page size 100; danh mục UUID loại văn bản chính thức
   loại bản dịch `BD` ngay phía server.
3. Detail và diagram gọi trực tiếp JSON API với concurrency 16.
4. Nếu detail thiếu content, spider gọi file-server action trong browser context; ưu tiên HTML,
   DOCX, PDF có digital text.
5. `Template.pdf` đúng 32.052 byte được nhận diện là placeholder upstream và quarantine, không OCR.
6. PDF scan được ghi `OCR_PENDING`; Tesseract chạy bằng worker riêng, không khóa network crawler.
7. Mỗi 1.000 record là một checkpoint: audit raw → gzip → audit lại. Resume bỏ shard đã pass.

Kafka và database được tách khỏi embedding/model:

- Canonical artifacts ưu tiên shard đã OCR-rescued.
- Kafka publisher dùng key `item_id`, idempotent producer, acks all và zstd.
- Raw archive consumer upsert manifest vào Neo4j bằng `MERGE`, commit Kafka sau DB success.
- Gate cuối so sánh không chỉ count mà cả tập ID và SHA-256 artifact–Kafka–database.

## 2. Kết quả raw crawl 8.000

| Trang | Record | Thời gian (giây) | Content | Upstream placeholder | OCR pending | Diagram valid |
|---:|---:|---:|---:|---:|---:|---:|
| 1–10 | 1.000 | 333,60 | 1.000 | 0 | 0 | 979 |
| 11–20 | 1.000 | 382,45 | 1.000 | 0 | 0 | 976 |
| 21–30 | 1.000 | 426,22 | 1.000 | 0 | 0 | 984 |
| 31–40 | 1.000 | 353,75 | 957 | 36 | 7 | 909 |
| 41–50 | 1.000 | 359,03 | 984 | 16 | 0 | 986 |
| 51–60 | 1.000 | 365,38 | 995 | 4 | 1 | 994 |
| 61–70 | 1.000 | 329,62 | 995 | 4 | 1 | 997 |
| 71–80 | 1.000 | 319,00 | 992 | 4 | 4 | 997 |

Tổng thời gian đóng tám shard là 2.869,04 giây, khoảng 47,82 phút; throughput trung bình
167,30 văn bản/phút. Raw gate:

- 8.000 record và 8.000 ID duy nhất.
- 0 duplicate xuyên shard.
- 7.923 content hợp lệ trước OCR worker.
- 64 upstream placeholder có fingerprint rõ ràng.
- 13 OCR pending trước worker; không có HTML rỗng không giải thích được.
- Artifact gzip gốc khoảng 72,83 MiB.
- `full_crawl_gate.json`: `passed=true`.

## 3. Hạn chế quan sát được

1. **Nguồn không có content:** 64 record có file trắng `Template.pdf` 32.052 byte. Đây là thiếu dữ
   liệu upstream, không phải OCR/crawler failure; record vẫn giữ metadata và diagram trong quarantine.
2. **PDF scan:** OCR inline từng làm throughput rơi từ khoảng 144 xuống 2 item/phút. Tách worker
   đưa network crawler trở lại khoảng 157–214 item/phút ở các phút cuối shard khó.
3. **Legacy VBPL:** endpoint trang cũ có thể được search engine lập chỉ mục nhưng trả 403 cho client;
   không dùng kỹ thuật giả user-agent/vượt WAF. Chỉ dùng API/file action công khai hiện hành.
4. **Dữ liệu nguồn thay đổi:** crawl theo trang trong nhiều giờ có rủi ro record mới làm dịch trang.
   Gate duplicate xuyên shard phát hiện trùng; full production nên lưu discovery manifest ID trước.
5. **Rate limit:** lượt 8.000 không ghi nhận HTTP 403/429 trong shard gate, nhưng full 160.660 vẫn
   phải theo dõi và hạ concurrency nếu nguồn throttling.
6. **Không đánh đồng metadata với content:** raw record/quarantine vẫn được đếm trong corpus 8.000,
   nhưng chỉ `html_status=VALID` mới được phép đi vào embedding sau này.

## 4. Thời gian dự kiến cho toàn bộ nguồn

Nguồn quan sát khoảng 160.660 record. Theo throughput thực đo 167,30 văn bản/phút:

```text
160.660 / 167,30 = 960,3 phút ≈ 16,0 giờ raw crawl
```

Nên lập kế hoạch 16–20 giờ để có biên cho throttling, checkpoint, retry và ghi đĩa. OCR là pha
riêng; thời gian phụ thuộc số/trang PDF scan và không được cộng vào đường nóng raw crawl.

## 5. Chứng minh 8.000 record là đúng bằng artifact, Kafka và database

Ba tầng bằng chứng bắt buộc:

1. **Artifact:** tám shard × 1.000, 8.000 unique ID, 0 duplicate, full raw gate pass.
2. **Kafka:** topic kiểm định mới có đúng 8.000 message/key; consumer group lag 0.
3. **Database:** Neo4j có đúng 8.000 node `RawLawDocument` cho
   `dataset_id=v6-8000-gzip`.

Verifier đọc lại toàn Kafka từ offset 0, lấy toàn bộ node database và so với canonical artifacts.
Count bằng nhau chưa đủ: `artifact_kafka_id_match`, `artifact_database_id_match`,
`artifact_kafka_hash_match`, `artifact_database_hash_match` đều phải `true`.

Kết quả thực tế trên topic `law-documents-v6-8000-gzip`:

- Publisher delivery: 8.000/8.000.
- Raw consumer: consumed 8.000, stored 8.000 trong 21,267 giây (`376,176 doc/s`).
- Neo4j mở trực tiếp bằng `cypher-shell`: 8.000 document, 8.000 distinct `doc_id`; trạng thái
  7.936 `VALID`, 64 `EMPTY` upstream placeholder.
- Kafka group `raw-archive-v6-8000-gzip`: `TOTAL-LAG=0` trên 8 partition; tổng high watermark
  là 8.000.
- Artifact/Kafka ID match: `true`; artifact/database ID match: `true`.
- Artifact/Kafka SHA-256 match: `true`; artifact/database SHA-256 match: `true`.
- `pipeline_8000_gate.json`: `passed=true`.

Kafka envelope gzip giải quyết payload HTML verbose: message lớn nhất ban đầu 12.430.821 byte,
sau envelope còn 541.757 byte; trung bình 13.625 byte và không có message vượt 950 KB.

Bằng chứng máy đọc nằm tại
`law_dataset/artifacts/full_crawl_v5/pipeline_8000_gate.json`; quy trình tái lập nằm trong
`docs/ubuntu_to_full_crawl_guide.md`.
