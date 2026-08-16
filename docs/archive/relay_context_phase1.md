# VietLawBERT — Handoff Context Phase 1

> Tài liệu chuyển tiếp cho cuộc trò chuyện mới. Cập nhật: **15/08/2026**.

## 1. Mục tiêu dự án

Xây dựng pipeline thu thập văn bản pháp luật Việt Nam từ [vbpl.vn](https://vbpl.vn/), xử lý cấu trúc pháp lý, tạo dữ liệu cho:

- Huấn luyện hoặc tiếp tục huấn luyện mô hình ngôn ngữ pháp luật kiểu BERT.
- Semantic search và RAG.
- Knowledge Graph về quan hệ sửa đổi, bổ sung, thay thế giữa văn bản.

Pipeline hiện dùng Scrapy + Playwright, Redpanda/Kafka, Milvus, Neo4j và MongoDB.

## 2. Môi trường hiện tại

- Host: Windows.
- Linux: Ubuntu 24.04 trên WSL 2.
- Repo Windows: `D:\vietlawbert`.
- Repo trong WSL: `/mnt/d/vietlawbert`.
- Python virtual environment: `~/.venvs/vietlawbert`.
- Docker Compose: `/mnt/d/vietlawbert/docker-compose.yml`.
- Docker data đã chuyển sang ổ D:
  `D:\Docker\wsl\DockerDesktopWSL\disk\docker_data.vhdx`.
- Ổ D còn khoảng 45 GB tại thời điểm bàn giao.
- Không dùng full CUDA stack; pilot dùng CPU-only.

## 3. Dịch vụ Docker

Các service đã chạy được:

| Service | Vai trò | Port chính |
|---|---|---:|
| `redpanda_vietlaw` | Kafka-compatible message broker | `9092` |
| `milvus_vietlaw` | Vector database | `19530`, `9091` |
| `neo4j_vietlaw` | Knowledge Graph | `7474`, `7687` |
| `mongodb_vietlaw` | Document/raw data database | `27017` |
| `milvus_etcd` | Milvus metadata | `2379-2380` |
| `milvus_minio` | Milvus object storage | `9001` |

Lệnh kiểm tra:

```bash
docker compose -f /mnt/d/vietlawbert/docker-compose.yml ps
```

## 4. Công việc đã hoàn thành

### 4.1. WSL và Docker

- Cài và dùng Ubuntu 24.04 trên Windows qua WSL 2.
- Di chuyển dữ liệu Docker sang ổ D để giảm áp lực cho ổ C.
- Xác nhận các container chính khởi động và hoạt động.
- Tránh xóa thủ công file VHDX của WSL/Docker.

### 4.2. Scrapy + Playwright

Đã cập nhật `requirements.txt`:

- Thêm `scrapy-playwright`.
- Thêm `playwright`.

Đã cập nhật `Dockerfile`:

- Cài Chromium bằng `playwright install --with-deps chromium`.

Đã cập nhật `law_dataset/src/crawler/settings.py`:

- Dùng Playwright download handler cho HTTP/HTTPS.
- Dùng `AsyncioSelectorReactor`.
- Tạo browser context `vbpl`.
- Cấu hình user agent, viewport, locale `vi-VN`, timezone `Asia/Ho_Chi_Minh`.
- Đưa log trên POSIX/WSL vào `~/.cache/vietlawbert/logs`.

Đã cập nhật `law_dataset/src/crawler/spiders/law_spider.py`:

- Bắt giá trị `next-action` từ request của browser thay vì hard-code.
- Gọi search Fetch trong browser context.
- Crawl trang chi tiết văn bản.
- Crawl diagram/API quan hệ văn bản.
- Giảm concurrency và thêm xử lý HTTP `403` để phù hợp hành vi của `vbpl.vn`.

Đã kiểm tra Chromium:

```text
Chromium OK
```

### 4.3. Crawl pilot

Đã crawl thành công 10 văn bản thử nghiệm:

- `17/2026/NĐ-CP`
- `80/2026/NĐ-CP`
- `81/2026/NĐ-CP`
- `99/2026/NĐ-CP`
- `122/2026/NĐ-CP`
- `123/2026/NĐ-CP`
- `135/2026/NĐ-CP`
- `238/2026/NĐ-CP`
- `236/2026/NĐ-CP`
- `241/2026/NĐ-CP`

Redpanda/Kafka đã nhận item.

### 4.4. Dependencies và model

Đã cài hoặc kiểm tra các thành phần chính:

- `torch==2.13.0+cpu`
- `transformers`
- `pymilvus`
- `neo4j`
- `openai`
- `confluent_kafka`
- Model embedding `BAAI/bge-m3`

Không chạy Ollama. `contextualizer.py` có fallback về text gốc khi LLM offline để ingestion không bắt buộc phải có LLM.

### 4.5. Làm sạch văn bản

Đã cập nhật `law_dataset/src/preprocess/text_cleaner.py`:

- Bổ sung pattern loại boilerplate quốc hiệu/cơ quan ban hành.
- Bổ sung pattern cho dạng Markdown table/dòng gạch ngang sinh ra từ HTML.

Chưa crawl lại toàn bộ pilot để xác nhận chất lượng sau thay đổi này.

### 4.6. Milvus

Đã cập nhật `law_dataset/src/database/milvus_client.py` để không tự động drop collection khi collection đã tồn tại; code dùng lại schema hiện có.

Điểm cần kiểm tra tiếp: replay Kafka phải không tạo duplicate vector. Khi ingestion đã idempotent, nên dùng `upsert` hoặc cơ chế ghi theo `chunk_id` ổn định.

### 4.7. Consumer

Đã chỉnh `law_dataset/src/streaming/consumer.py` để thử nghiệm:

- Tắt auto commit.
- Commit theo message sau xử lý.
- Giảm batch size để debug.
- Thêm timeout session.

Tuy nhiên, code hiện tại còn điểm cần sửa trước khi dùng production:

- `group.id` đang tạo UUID mới mỗi lần chạy (`vietlawbert-consumers-<uuid>`), khiến consumer đọc lại từ đầu và làm replay.
- Cần dùng group ID ổn định.
- Chỉ commit sau khi cả Milvus và Neo4j xử lý thành công.
- Cần retry hoặc dead-letter record khi batch thất bại.
- Cần bảo đảm Milvus dùng thao tác idempotent khi Kafka replay.

## 5. Trạng thái database gần nhất

Đã chạy `law_dataset/src/inspect_db.py` trong WSL.

### Milvus

```text
Trạng thái: HOẠT ĐỘNG
Tổng số đoạn luật: 1038
```

### Neo4j

```text
LawDocument: 55
Chapter: 14
Article: 85
Chunk: 888
Edges: 1161
```

Lệnh kiểm tra:

```bash
cd /mnt/d/vietlawbert
source ~/.venvs/vietlawbert/bin/activate
export PYTHONPATH=/mnt/d/vietlawbert/law_dataset/src:$PYTHONPATH
python law_dataset/src/inspect_db.py
```

## 6. Quality gate chưa đạt

Chưa được full crawl hoặc train model. Các vấn đề chính:

### 6.1. Lệch số chunk

Milvus có `1038` chunk, Neo4j có `888` chunk.

Nguyên nhân cần xác minh:

- Consumer từng lỗi giữa lúc ghi Milvus và Neo4j.
- Kafka replay nhiều lần.
- Offset/group ID không ổn định.
- Hai database chưa idempotent hoàn toàn.

### 6.2. Quan hệ Neo4j bị trùng

Một quan hệ mẫu xuất hiện nhiều lần:

```text
[Nghị định 17/2026/NĐ-CP]
  --(VAN_BAN_SUA_DOI_BO_SUNG)-->
[Luật Hàng không dân dụng Việt Nam số 66/2006/QH11]
```

Code cũ dùng `apoc.create.relationship`, luôn tạo relationship mới. Hướng sửa đúng là dùng Cypher `MERGE` với relationship type đã sanitize, thay vì tạo relationship động không idempotent.

Đã có ý định chuyển sang `apoc.merge.relationship`, nhưng chưa xác minh cú pháp và kết quả thực tế. Ưu tiên dùng Cypher `MERGE` trực tiếp.

Query kiểm tra duplicate:

```cypher
MATCH (a:LawDocument)-[r]->(b:LawDocument)
RETURN a.doc_id, type(r), b.doc_id, count(r)
ORDER BY count(r) DESC
LIMIT 20
```

### 6.3. Boilerplate trong chunk

Sample cũ còn nội dung:

```text
CHÍNH PHỦ | CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
```

Pattern đã bổ sung vào `text_cleaner.py`, nhưng cần crawl/ingest lại rồi kiểm tra sample mới.

### 6.4. Hierarchy bị null

Một sample chunk có metadata:

```json
{
  "phần": null,
  "chương": null,
  "mục": null,
  "tiểu_mục": null,
  "điều": null,
  "khoản": null,
  "điểm": null
}
```

Có thể đây là chunk phần mở đầu. Cần kiểm tra nhiều chunk. Nếu phần lớn metadata null, phải sửa parser Markdown/HTML trước khi mở rộng dữ liệu.

## 7. Reset pilot

Reset sẽ xóa dữ liệu pilot hiện tại. Chỉ reset khi dữ liệu hiện tại chỉ là dữ liệu thử nghiệm và đã xác nhận không cần giữ.

Lệnh inline Neo4j từng lỗi vì Bash history expansion:

```text
-bash: !': event not found
```

Nguyên nhân là chuỗi `print('Neo4j cleaned!')` chứa dấu `!` trong Bash.

Script reset chạy từ Windows cũng từng lỗi vì Python Windows không có `pymilvus`. Reset phải chạy trong Ubuntu WSL, sau khi Docker đang chạy và venv đã activate.

Không dùng lệnh xóa database nếu chưa xác nhận rõ dữ liệu cần giữ.

## 8. Việc cần làm tiếp theo

### Bước 1 — Kiểm tra syntax

Chạy trong Ubuntu WSL:

```bash
cd /mnt/d/vietlawbert
source ~/.venvs/vietlawbert/bin/activate
export PYTHONPATH=/mnt/d/vietlawbert/law_dataset/src:$PYTHONPATH

python -m py_compile law_dataset/src/preprocess/text_cleaner.py
python -m py_compile law_dataset/src/database/neo4j_client.py
python -m py_compile law_dataset/src/database/milvus_client.py
python -m py_compile law_dataset/src/streaming/consumer.py
```

### Bước 2 — Sửa idempotency

- Dùng Kafka group ID ổn định.
- Không commit message trước khi xử lý thành công.
- Commit đúng message đã xử lý.
- Retry lỗi tạm thời.
- Ghi dead-letter cho item lỗi lâu dài.
- Dùng `upsert` hoặc unique key theo `chunk_id` trong Milvus.
- Dùng `MERGE` cho node và relationship Neo4j.

### Bước 3 — Reset pilot có kiểm soát

- Xóa collection Milvus pilot.
- Xóa graph Neo4j pilot.
- Không xóa Docker volume nếu chưa cần.
- Crawl lại đúng 10 văn bản pilot.
- Chạy consumer với group ID ổn định.
- Chạy lại `inspect_db.py`.

### Bước 4 — Chạy quality gate

Cần đạt tối thiểu:

- Số chunk giữa Milvus và Neo4j khớp hoặc có sai lệch được giải thích.
- Không còn duplicate relationship.
- Boilerplate giảm về mức chấp nhận được.
- Tỷ lệ chunk có hierarchy hợp lệ đủ cao.
- Consumer không mất message khi restart.
- Kafka offset không replay ngoài chủ ý.
- LLM offline không làm fail ingestion.

### Bước 5 — Mở rộng crawl

Chỉ sau khi quality gate đạt:

- Mở rộng theo năm, loại văn bản và cơ quan ban hành.
- Lưu raw response, metadata, checksum và thời điểm crawl.
- Thêm checkpoint/resume.
- Thêm rate limit, retry có backoff và theo dõi lỗi `403/429/5xx`.
- Chạy kiểm tra chất lượng theo batch.

### Bước 6 — Chuẩn bị corpus/model

- Xuất corpus sạch theo document/chapter/article/chunk.
- Tách train/validation/test theo văn bản, tránh rò rỉ các phiên bản cùng văn bản.
- Xây benchmark truy hồi và bài toán pháp lý tiếng Việt.
- Chỉ bắt đầu train sau khi dữ liệu và quality gate ổn định.

## 9. Trạng thái thật tại thời điểm bàn giao

```text
Playwright setup: DONE
WSL/Docker relocation: DONE
VBPL probe: DONE
10-document crawl: DONE
Kafka ingestion: PARTIAL
Milvus: RUNNING, data exists
Neo4j: RUNNING, data exists
Consumer: RUNNING PARTIAL, offset/idempotency cần verify
Quality gate: NOT PASSED
Full crawl: NOT STARTED
Training corpus: NOT READY
Model training: NOT STARTED
```

## 10. Tin nhắn khởi đầu cho cuộc trò chuyện mới

Có thể dán đoạn sau vào cuộc trò chuyện mới:

```text
Đây là context dự án VietLawBERT. Repo Windows: D:\\vietlawbert; repo WSL: /mnt/d/vietlawbert; Ubuntu 24.04 WSL2; venv: ~/.venvs/vietlawbert. Đã hoàn thành Playwright setup, crawl pilot 10 văn bản vbpl.vn, Docker/Milvus/Neo4j/Redpanda/MongoDB đang chạy. DB gần nhất: Milvus 1038 chunks; Neo4j 55 LawDocument, 14 Chapter, 85 Article, 888 Chunk, 1161 edges. Quality gate chưa đạt vì lệch chunk, duplicate relationship Neo4j, boilerplate và hierarchy null cần kiểm tra. Chưa full crawl, chưa train model. Đọc file D:\\vietlawbert\\docs\\relay_context_phase1.md rồi tiếp tục từ bước kiểm tra syntax, sửa consumer idempotency và Neo4j MERGE; không full crawl trước khi quality gate đạt.
```
