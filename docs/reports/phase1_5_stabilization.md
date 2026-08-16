# VietLawBERT - Phase 1.5: Stabilization & Quality Gate
*Ngày cập nhật: 16/08/2026*

## 1. Tóm tắt các vấn đề đã giải quyết

### 1.1. Lỗi Idempotency của Kafka Consumer
- **Vấn đề:** Consumer dùng `uuid4()` để tạo `group.id` mới mỗi lần chạy. Hệ quả: Mỗi lần khởi động lại, Consumer đọc lại Kafka từ đầu, gây ra replay message chồng chéo.
- **Giải pháp:** Cố định `KAFKA_GROUP_ID` (ví dụ: `vietlawbert-consumers-batch-50`). Sửa logic chỉ commit Kafka offset khi toàn bộ tiến trình (Milvus, Neo4j) đã lưu thành công.

### 1.2. Lỗi Duplicate trong Milvus và Neo4j
- **Vấn đề Milvus:** Dùng hàm `client.insert()`. Khi Kafka replay, Milvus nạp chồng các Chunk ID cũ thành các vector mới.
- **Giải pháp Milvus:** Chuyển sang `client.upsert()`. Dựa vào Primary Key `chunk_id`, nếu đã có thì ghi đè, nếu chưa có thì thêm mới.
- **Vấn đề Neo4j:** Hàm `apoc.create.relationship()` tạo ra vô số mũi tên quan hệ hệt nhau giữa 2 node nếu chạy lại nhiều lần.
- **Giải pháp Neo4j:** Chuyển sang dùng Cypher `MERGE (s)-[rel:EDGE_TYPE]->(t)`.

### 1.3. Lỗi Text Preprocessing (Quality Gate)
- **Vấn đề Boilerplate:** Các câu như "CHÍNH PHỦ - CỘNG HÒA XÃ HỘI CHỦ NGHĨA VN" và các đường gạch ngang Markdown `---` còn sót trong chunk.
- **Vấn đề Hierarchy:** Phần mở đầu văn bản (Preamble) luôn có cấu trúc rỗng `{"phần": null, "điều": null...}`.
- **Giải pháp:** Vá regex trong `text_cleaner.py` và sửa `legal_chunker.py` để bổ sung default value `{"phần": "Phần Mở đầu"}`.

### 1.4. Lỗi Rate Limit khi gọi LLM Contextualizer (API 429)
- **Vấn đề:** Groq Cloud giới hạn 30 requests/min. Một văn bản bị cắt thành gần 200 chunks, tạo ra 200 API calls trong 2 giây làm hệ thống bị Block (Lỗi 429 Too Many Requests).
- **Giải pháp:** Tạm vô hiệu hoá LLM Text Contextualization. Hệ thống dùng trực tiếp text gốc để ingest vào DB. LLM chỉ nên dùng cho các tác vụ khó như OCR PDF trong tương lai.

### 1.5. Lỗi Treo Crawler
- **Vấn đề:** Terminal chạy `scrapy` trắng xóa do cấu hình tắt `LOG_STDOUT`. Spider phớt lờ tham số `pages=5`.
- **Giải pháp:** Bật lại Log ra màn hình console và vá hàm `__init__` để tiếp nhận giới hạn page.

---

## 2. Kết quả Quality Gate

Thành công. Hệ thống đồng nhất 211 chunks trên cả Milvus và Neo4j, không còn quan hệ rác, không còn boilerplate.

---

## 3. Road Map Phase 2 (Kiến trúc Production)

Để giải quyết bài toán cào và nhúng (embedding) hơn 100,000 văn bản pháp luật, hệ thống cần nâng cấp theo hướng:

1. **Quản trị Tiến trình (Daemon/Background)**
   - Không chạy Consumer bằng terminal trực tiếp dễ đứt gãy.
   - Dùng `PM2` hoặc `Systemd` để chạy nền, tự động auto-restart và quản lý log tập trung.
2. **Offload Embedding (Đưa Vector hóa lên API Cloud)**
   - Việc `consumer.py` ôm đồm load model `BAAI/bge-m3` vào RAM (CPU) quá chậm (có thể bị Kafka kích khỏi group vì timeout `max.poll.interval.ms`).
   - Sẽ chuyển việc sinh 1024-dim Vector sang dùng dịch vụ API Embedding ngoài (như NVIDIA NIM, Nomic) để hoàn thành trong mili-giây.
3. **Nâng cấp Crawler (Từ Playwright sang API-First)**
   - Mở Chrome/Playwright cào Web là quá tốn tài nguyên. Chuyển sang tìm Endpoint API ngầm của VBPL, lấy Data Json/Raw trực tiếp qua HTTP.
   - Chạy song song (Concurrency) 50 luồng, sử dụng Proxy Rotation để chống khoá IP.

---

## 4. Phase 1.6 - Consumer Throughput & Pilot-100 Readiness (16/08/2026)

### 4.1. Nguyên nhân consumer chậm/giống bị treo

- Batch cũ tính theo số message/văn bản, nên 5 văn bản có thể gom hàng trăm chunk trước khi ghi.
- Batch cuối dưới 5 message không flush khi Kafka tạm hết dữ liệu.
- Lỗi xử lý message bị bắt rồi bỏ qua, tạo nguy cơ commit offset của message sau vượt qua message lỗi.
- Benchmark trực tiếp trên Ubuntu: BGE-M3 CPU FP32 đạt `1,26 chunk/giây` với batch ghi 64 chunk. Đây là bottleneck tính toán thật, không phải Kafka đứng.

### 4.2. Thay đổi đã triển khai

- `consumer.py`: giới hạn batch ghi theo chunk (`64` mặc định), idle flush 5 giây, idle exit cho batch job, log chunk/giây, fail-fast khi record lỗi và commit offset theo từng Kafka partition.
- `milvus_client.py`: tự chọn CPU/CUDA, batch cấu hình qua env, length-bucketing giảm padding, tùy chọn CPU INT8 và embedding API OpenAI-compatible 1024 chiều.
- Kết quả CPU INT8 + length-bucketing: `3,04 chunk/giây`, nhanh hơn khoảng `2,4 lần` so với phép đo FP32 nói trên.
- `law_spider.py`: giới hạn chính xác `limit`, hỗ trợ `pages/page_size/keyword/agency_ids`, search pagination nằm trong cùng Playwright context, concurrency API mặc định tăng từ 1 lên 4.
- `pipelines.py`: Kafka broker/topic từ env, idempotent producer, delivery callback và không nuốt lỗi delivery.
- `legal_chunker.py`: nhận tiêu đề Markdown dạng `**“Điều 31...` và không merge preamble/khoản/điểm qua ranh giới cấu trúc.
- `audit_pilot.py`: so trực tiếp `chunk_id` Milvus–Neo4j, kiểm tra duplicate relation, hierarchy, boilerplate, HTML và document count.

### 4.3. Kết quả xác minh

- Unit tests stabilization: `9/9` pass, gồm commit đa partition và quarantine relation type chưa xác định.
- Smoke pagination: `11/11` item qua 2 trang search, 0 HTTP error, 0 duplicate, 0 HTML rỗng, hoàn thành trong `23,08 giây` (`28,70 item/phút`). Có `7/11` diagram `INCONSISTENT` do mã nhóm quan hệ chưa được mapping; các cạnh này đã được quarantine.
- Audit dữ liệu cũ theo tập ID: Milvus `803`, Neo4j `803`, sai lệch `0`, duplicate relation `0`, boilerplate `0`; còn `1` hierarchy mismatch và chỉ có `9` source document.
- Topic pilot mới là `law-documents-v3`, tách khỏi backlog và smoke message thử nghiệm cũ.

### 4.4. Quyết định readiness

- **Crawler nội dung pilot 100:** sẵn sàng chạy theo `docs/guides/pilot_100_runbook.md`; pagination và giới hạn item đã được kiểm chứng thật.
- **Ontology/diagram pilot 100:** chưa pass. Pilot 100 phải dùng để thống kê các mã quan hệ, xác minh mapping rồi crawl/ingest lại trước full crawl.
- **Consumer pilot 100:** sẵn sàng về tính đúng/idempotency; CPU INT8 dùng được cho pilot nhưng cần chấp nhận thời gian hàng chục phút.
- **Full crawl + full embedding:** chưa được phép mở. Cần pilot 100 pass toàn bộ quality gate, giải quyết record thiếu HTML bằng PDF/OCR và chuyển embedding sang GPU/service có throughput phù hợp.

### 4.5. Kết quả pilot 100 thực tế

- Crawl hoàn tất `100/100` văn bản trong `128,68 giây` (`46,88 item/phút`).
- Toàn bộ `201/201` HTTP response trả mã `200`; không có item thất bại.
- Artifact có 100 document ID duy nhất, `100/100` HTML hợp lệ, 0 duplicate, 0 thiếu ID.
- Diagram: 81 `EMPTY`, 19 `INCONSISTENT`, 0 `VALID`. Tám nhóm mã chưa mapping được ghi trong `law_dataset/artifacts/crawl_100_gate.json`.
- Consumer CPU INT8 được chạy thử 5 batch (`320` chunk), throughput quan sát `1,80–2,62 chunk/giây`. Audit phần ghi thử xác nhận Milvus `320` = Neo4j `320`, sai lệch ID `0`, hierarchy mismatch `0`, boilerplate `0`. Consumer được dừng trước khi commit; group v3 còn nguyên lag 100 nên có thể replay toàn bộ an toàn.
- Kết luận: crawl nội dung 100 đã đạt; graph ontology và throughput full corpus chưa đạt. Không mở full crawl ở trạng thái này.

### 4.6. Pilot chuẩn v5 và gate end-to-end cuối cùng

Kết quả v3 ở trên được giữ lại làm lịch sử phát hiện lỗi. Corpus chuẩn cuối dùng topic
`law-documents-v5`, group `vietlawbert-consumers-v5-bounded` và artifact
`law_dataset/artifacts/crawl_100_v5.jsonl`.

- Search gửi danh sách UUID `docType` chính thức để loại `BD` ngay trên server; không còn quét khoảng 20 bản dịch để lấy 1 văn bản gốc.
- Ảnh `data:` nhúng bị loại khỏi HTML, `documentContent.content` không bị lặp trong metadata và Kafka dùng nén zstd. Payload lớn nhất trong pilot là 387.356 byte, không còn lỗi `MSG_SIZE_TOO_LARGE`.
- Mapping 15 mã quan hệ được xác minh từ bundle công khai của vbpl.vn; nhóm target là `OUTGOING`, nhóm source là `INCOMING`. Quan hệ chưa biết vẫn bị quarantine.
- Crawl gate: 100 record/100 ID, 100 HTML hợp lệ, 0 bản dịch, 100 `docType.parentCode=VBQPPL`, 0 văn bản hành chính, 99 diagram `VALID`, 1 `EMPTY`, 0 key chưa mapping.
- Chunker thực thi thật giới hạn 1.600 ký tự (tham số cũ từng bị bỏ qua), giữ hierarchy khi buộc phải tách atomic unit quá dài.
- Consumer ghi 2.125 chunk trong khoảng 16 phút ingest, throughput end-to-end trung bình khoảng 2,20 chunk/giây trên CPU INT8. Offset cuối 100/100, lag 0 và idle-exit bình thường.
- Benchmark embedding thuần trên 128 chunk thật: batch 16 đạt 2,648 chunk/giây, batch 32 đạt 2,550 chunk/giây; giữ batch 16.
- Database gate: Milvus 2.125 = Neo4j 2.125; 100 source document; 0 ID lệch, duplicate relation, hierarchy mismatch, boilerplate, chunk rỗng hoặc chunk quá 1.600 ký tự.
- Toàn bộ stabilization test: 14/14 pass; compile Python và `docker compose config --quiet` pass.

### 4.7. Benchmark crawler và quyết định full crawl

| Profile | Kết quả | Throughput | So với pilot cũ |
|---|---:|---:|---:|
| page 10, concurrency 4, delay 0,5 s | 128,28 s / 100 | 46,41 doc/phút | 1,00x |
| page 100, concurrency 8, delay 0,1 s | ~46 s / 100 | 130,43 doc/phút | 2,81x |
| page 100, concurrency 16, delay 0 s | 33,55 s / 100 | 181,82 doc/phút | 3,92x |

Cả hai profile nhanh đều pass gate 100/100; profile concurrency 16 có 201/201 HTTP 200.
Nguồn hiện báo khoảng 160.660 record (16.066 trang ở page size 10). Ngoại suy profile nhanh
là khoảng 14,7 giờ crawl, nhưng đây chưa phải SLA chạy dài; full crawl phải checkpoint và hạ
concurrency nếu xuất hiện 403/429.

**Quyết định:** crawler và tính đúng của consumer đã sẵn sàng cho một đợt crawl theo checkpoint.
Không chạy full embedding bằng một CPU hiện tại: mật độ pilot 21,25 chunk/văn bản tương đương
khoảng 3,4 triệu chunk; ở 2,20 chunk/giây sẽ mất xấp xỉ 18 ngày liên tục. Trước full embedding
phải dùng GPU hoặc endpoint embedding 1024 chiều và benchmark retrieval/throughput. Các văn bản
cũ thiếu HTML cũng phải đi qua nhánh rescue PDF/OCR thay vì được commit như record hoàn chỉnh.

### 4.8. Checkpoint 1.000, rescue OCR và image triển khai

- Fast profile concurrency 16 hoàn tất 1.000 văn bản trong `307,35 giây`, tương đương
  `195,44 văn bản/phút`; `2.001/2.001` HTTP response trả mã 200.
- Gate ban đầu có 999/1.000 nội dung hợp lệ. Record duy nhất thiếu text là
  `92532ed0-76b1-11f1-bdd5-25feca9f8958` (`108/2026/QĐ-UBND`, Gia Lai); detail API khai báo
  có nội dung nhưng các bản HTML/DOCX/PDF đều là ảnh scan.
- Spider hiện tự gọi file-server bằng Playwright khi detail rỗng, ưu tiên HTML/DOCX/PDF, và OCR
  PDF bằng Tesseract `vie+eng` nếu không có digital text. Retry record trên hoàn tất trong
  `18,38 giây`, trạng thái `PDF_OCR_RECOVERED`/`OCR_COMPLETED`, thu được 6.324 ký tự.
- `merge_crawl_artifacts.py` overlay retry theo `item_id`, giữ thứ tự checkpoint và không cần
  crawl lại shard. Gate sau merge: 1.000/1.000 ID duy nhất và nội dung thật, 0 bản dịch,
  0 văn bản hành chính, 979 diagram `VALID`, 21 `EMPTY`, 0 `INCONSISTENT`, 0 relation key lạ.
- Audit không còn tin riêng cờ `html_status=VALID`: nội dung `html_raw` phải có tối thiểu
  100 ký tự, nhờ đó record gắn nhãn hợp lệ nhưng rỗng sẽ fail gate.
- Docker build context giảm xuống khoảng 68 KB bằng `.dockerignore`. Image crawler/consumer dùng
  PyTorch `2.13.0+cpu`, không kéo CUDA/NVIDIA hay oneAPI không dùng; image hoàn chỉnh 1,18 GB và
  có Tesseract 5.5 với `vie`, `eng`.
- Artifact 1.000 record có kích thước 67,53 MiB. Ngoại suy raw full corpus khoảng 10,59 GiB;
  ổ D còn 43,73 GiB tại thời điểm kiểm tra. Ở throughput checkpoint, raw crawl 160.660 record
  mất khoảng 13,7 giờ nếu tốc độ được giữ ổn định.
- Stabilization tests sau cùng: `23/23` pass.
- `audit_pilot.py` đã chuyển sang streaming để full artifact 10+ GiB không bị nạp hết vào RAM.
  `run_crawl_shards.py` chia mặc định 1.000 record/shard, audit raw, nén, audit gzip, lưu state,
  resume bằng cách skip shard đã pass và kiểm tra duplicate xuyên shard ở gate cuối. Smoke crawl
  mới và smoke resume đều pass; stabilization tests sau runner là `24/24`.

**Kết luận cập nhật:** crawler raw đã vượt gate 100 và 1.000, có checkpoint/retry/OCR đủ để mở
full crawl theo shard. Full embedding vẫn không nên chạy trên một CPU: cần GPU hoặc endpoint
1024 chiều trước khi phát toàn corpus vào consumer.

### 4.9. Chuyển từ kế hoạch full crawl sang gate 8.000

- Job ban đầu được cấu hình cho 160.660 document, nhưng đã dừng có kiểm soát khi phạm vi mục tiêu
  được chốt lại ở tối đa 8.000. Job sau đó được chạy lại với `--total-documents 8000`, nên runner
  không thể đi quá shard 8.
- Shard 1 (trang 1–10) pass: 1.000/1.000 nội dung, 979 diagram valid, không duplicate,
  translated, administrative hoặc inconsistent; gzip còn 8,45 MiB.
- Tiến độ và resume được đọc từ `law_dataset/artifacts/full_crawl_v5/crawl_state.json`; terminal
  yên lặng giữa hai checkpoint không được dùng làm dấu hiệu treo vì tiến độ chi tiết nằm trong
  log từng shard.
- Kết quả cuối của phạm vi này được xác nhận ở mục 4.12: đủ tám shard, đúng 8.000 ID duy nhất và
  `full_crawl_gate.json` có `passed=true`. Full crawl 160.660 chỉ còn là bước vận hành dự kiến.

### 4.10. Tách OCR khỏi đường nóng và gate quarantine

- Shard 4 lộ đúng dạng “đứng”: một PDF scan dài chạy Tesseract đồng bộ trong callback làm
  logstats từ khoảng 144 item/phút rơi xuống 2 item/phút dù không có lỗi mạng.
- PDF extraction/OCR đã chuyển sang worker thread, giới hạn `OCR_CONCURRENCY=2`. Quan trọng hơn,
  full runner có `--defer-ocr`: raw crawl ghi `OCR_PENDING`, còn targeted OCR chạy ở pha riêng.
- Browser rescue đóng page trước retry, chỉ chờ `domcontentloaded`, chặn image/font/media/CSS/script;
  targeted retry 9 ID không còn `Page.goto` timeout.
- File trắng upstream được nhận diện bằng fingerprint `Template.pdf` + 32.052 byte và ghi
  `UPSTREAM_TEMPLATE`, tránh tải/render/OCR lặp lại. Smoke 4 placeholder giảm từ khoảng 17–18 giây
  xuống 7,45 giây.
- Audit vẫn strict mặc định. Full mode chỉ cho phép hai quarantine có bằng chứng:
  `UPSTREAM_TEMPLATE` khi API `hasContent=false`, hoặc `OCR_PENDING`; HTML rỗng khác vẫn fail.
- Kết quả shard 4 sau tách pha: 1.000/1.000 ID, 957 content hợp lệ, 36 upstream placeholder,
  7 OCR pending, 0 invalid không giải thích được, 0 duplicate/bản dịch/hành chính/diagram
  inconsistent. Throughput các phút cuối đạt 157, 199 và 214 item/phút; job chuyển shard 5.
- Mỗi shard sinh `*.quarantine.jsonl` riêng để targeted retry/overlay sau raw crawl.
- Test suite hiện `31/31` pass; compile và compose config pass.

### 4.11. OCR worker incremental

- `run_ocr_quarantine.py` đọc các file `*.quarantine.jsonl`, chọn riêng `OCR_PENDING`, chia batch
  targeted, bật OCR inline trong worker riêng và không phát Kafka.
- Chỉ record OCR thành công được overlay; base gzip không bị sửa. Worker tạo bản
  `*.rescued.jsonl.gz`, file recovered/unresolved và `ocr_retries/ocr_state.json`.
- Lượt chạy sau bảo toàn state và bỏ shard đã xử lý, phù hợp chạy incremental song song với raw
  crawler.
- Kiểm thử thật: shard 4 phục hồi 7/7 scan (0 unresolved), content tăng 957 → 964 và OCR pending
  giảm 7 → 0; shard 6 phục hồi 1/1, content tăng 995 → 996. Cả hai rescued gate pass.
- Trong khi OCR worker chạy PDF dài 4.398–52.311 ký tự, raw crawler vẫn tiến triển sang shard 7;
  điều này xác minh cách ly tải CPU khỏi đường network.
- Test suite sau OCR runner: `32/32` pass.

### 4.12. Gate cuối giới hạn 8.000: Kafka và database

- Theo mục tiêu cập nhật, runner được dừng ở đúng 8 shard/8.000 record; không crawl shard 9.
- Raw gate: 8.000 ID duy nhất, 0 duplicate xuyên shard, 7.923 content trước OCR, 64 upstream
  placeholder, 13 OCR pending và 0 invalid không giải thích được.
- OCR worker phục hồi 13/13 scan, 0 unresolved. Canonical gate: 7.936 content hợp lệ,
  64 upstream placeholder, OCR pending 0; tám shard đều pass.
- Tổng thời gian tám shard là 2.869,04 giây (47,82 phút), trung bình 167,30 doc/phút. Ngoại suy
  160.660 record là khoảng 16,0 giờ raw crawl; lập kế hoạch vận hành 16–20 giờ.
- HTML nguồn có payload lớn nhất 12.430.821 byte. Kafka gzip envelope kèm SHA-256 giảm message lớn
  nhất còn 541.757 byte, trung bình 13.625 byte; 0/8.000 vượt 950 KB.
- Topic chuẩn `law-documents-v6-8000-gzip`: producer delivery 8.000/8.000.
- Raw archive consumer không embedding: consumed 8.000, stored Neo4j 8.000 trong 21,267 giây,
  đạt 376,176 doc/s; group `raw-archive-v6-8000-gzip` có `TOTAL-LAG=0`.
- Mở Neo4j bằng `cypher-shell`: 8.000 node và 8.000 distinct `doc_id`; 7.936 `VALID`, 64 `EMPTY`.
- `pipeline_8000_gate.json`: artifact = Kafka = database đều 8.000; tập ID và SHA-256 khớp ở
  cả bốn phép so; `passed=true`.
- Test suite cuối: `34/34` pass. Không chạy train/model trong phase này.
