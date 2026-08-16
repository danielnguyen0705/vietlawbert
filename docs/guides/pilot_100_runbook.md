# Runbook pilot 100 và full crawl VietLawBERT

*Cập nhật: 16/08/2026 — baseline v5 đã pass end-to-end.*

## 1. Baseline đã xác minh

- Topic: `law-documents-v5`; consumer group: `vietlawbert-consumers-v5-bounded`.
- Artifact: `law_dataset/artifacts/crawl_100_v5.jsonl`.
- Crawl gate: 100/100 ID, 100 HTML, 0 bản dịch, 0 hành chính, 0 relation key chưa mapping.
- Ingestion gate: 2.125 Milvus chunk = 2.125 Neo4j chunk, 100 source document, lag Kafka 0.
- Không có duplicate relation, hierarchy mismatch, boilerplate, chunk rỗng hoặc chunk >1.600 ký tự.
- CPU INT8 end-to-end: khoảng 2,20 chunk/giây; pilot ingest khoảng 16 phút.
- Embedding thuần 128 chunk: batch 16 đạt 2,648 chunk/giây, tốt hơn batch 32 ở 2,550 chunk/giây.
- Crawler profile 8 đạt 130,43 doc/phút; profile 16 đạt 181,82 doc/phút trên pilot 100.
- Checkpoint 1.000 đạt 195,44 doc/phút; sau OCR retry, 1.000/1.000 nội dung pass gate.

## 2. Preflight

```bash
cd /mnt/d/vietlawbert
source ~/.venvs/vietlawbert/bin/activate
export PYTHONPATH=/mnt/d/vietlawbert/law_dataset/src

python -m unittest tests.test_ingestion_stabilization -v
python -m pytest tests -q
python -m compileall -q law_dataset/src
docker compose config --quiet
docker compose up -d redpanda neo4j etcd minio milvus-standalone
```

Không trộn model/provider embedding khác nhau trong cùng collection. Có thể đặt
`MILVUS_COLLECTION` để tạo collection version mới khi thay model hoặc schema.

## 3. Crawl pilot hoặc checkpoint

Profile mặc định an toàn đã được đặt ở concurrency 8, delay 0,1 giây:

```bash
cd /mnt/d/vietlawbert/law_dataset/src
export KAFKA_BROKER=localhost:9092
export KAFKA_TOPIC=law-documents-v5

scrapy crawl law_spider \
  -a limit=100 \
  -a page_size=100 \
  -O ../artifacts/crawl_100_v5.jsonl
```

Fast profile đã pass 100/100 nhưng phải theo dõi 403/429 khi chạy dài:

```bash
export CRAWLER_CONCURRENCY=16
export CRAWLER_DOWNLOAD_DELAY=0
```

Đặt `KAFKA_ENABLED=0` để benchmark/audit artifact mà không phát message. Spider mặc định loại
`docType=BD` ngay trên server và vẫn lọc phòng vệ ở client. Không dùng `isLw` làm tiêu chí pháp
luật; trường này không đồng nghĩa với VBQPPL. Gate dùng `docType.parentCode` và
`isAdministrativeDocument`.

## 4. Consumer

CPU INT8 (chỉ phù hợp pilot):

```bash
cd /mnt/d/vietlawbert
source ~/.venvs/vietlawbert/bin/activate
export PYTHONPATH=/mnt/d/vietlawbert/law_dataset/src
export KAFKA_BROKER=localhost:9092
export KAFKA_TOPIC=law-documents-v5
export KAFKA_GROUP_ID=vietlawbert-consumers-v5-bounded
export CONSUMER_DOC_BATCH_SIZE=10
export CONSUMER_CHUNK_BATCH_SIZE=64
export CONSUMER_FLUSH_INTERVAL_SECONDS=5
export EMBED_PROVIDER=local
export EMBED_DEVICE=cpu
export EMBED_BATCH_SIZE=16
export EMBED_CPU_INT8=1

cd law_dataset/src
python -m ingestion.embedding_consumer --idle-exit-seconds 30
```

Production phải dùng GPU hoặc OpenAI-compatible endpoint trả đúng vector 1024 chiều:

```bash
export EMBED_PROVIDER=openai_compatible
export EMBED_API_BASE=http://embedding-server:8000/v1
export EMBED_API_KEY=...
export EMBED_API_MODEL=...
```

Offset chỉ commit sau khi Milvus, Neo4j và semantic relations đều thành công. Batch cuối được
idle-flush; lỗi record làm process fail để Kafka replay thay vì bỏ qua.

## 5. Quality gate

```bash
cd /mnt/d/vietlawbert
python law_dataset/src/audit_pilot.py \
  --crawl-file law_dataset/artifacts/crawl_100_v5.jsonl \
  --databases \
  --expect-documents 100 \
  --output law_dataset/artifacts/ingestion_100_v5_gate.json

docker exec redpanda_vietlaw \
  rpk group describe vietlawbert-consumers-v5-bounded
```

Chỉ tiếp tục khi report có `"passed": true` và Kafka `TOTAL-LAG=0`.

### Retry, OCR và ghép checkpoint

Record thiếu content không được phát Kafka mặc định. Image Docker có Tesseract `vie+eng`; retry
chính xác các ID lỗi mà không crawl lại shard:

```bash
docker run --rm \
  -v /mnt/d/vietlawbert:/app \
  -w /app/law_dataset/src \
  -e KAFKA_ENABLED=0 \
  -e OCR_ENABLED=1 \
  -e OCR_LANG=vie+eng \
  vietlawbert-app \
  scrapy crawl law_spider \
    -a doc_ids=ID_1,ID_2 \
    -O ../artifacts/retry.jsonl

python law_dataset/src/audit_pilot.py \
  --crawl-file law_dataset/artifacts/retry.jsonl \
  --expect-documents 2

python law_dataset/src/merge_crawl_artifacts.py \
  --base law_dataset/artifacts/checkpoint.jsonl \
  --overlay law_dataset/artifacts/retry.jsonl \
  --output law_dataset/artifacts/checkpoint_rescued.jsonl
```

Chỉ thay base bằng file đã rescue sau khi cả retry và file sau merge đều pass audit.

## 6. Full crawl theo checkpoint

Nguồn hiện khoảng 160.660 record. Checkpoint 1.000 thực tế mất 307,35 giây (195,44 doc/phút),
toàn bộ HTTP 200 và pass sau một OCR retry. Không chạy một job đơn khối không checkpoint.
Runner chuẩn chia 10 trang/1.000 record, audit từng JSONL, nén streaming, audit lại file gzip,
ghi `crawl_state.json`, và tự skip shard đã pass khi resume:

```bash
docker run --rm \
  -v /mnt/d/vietlawbert:/app \
  -w /app/law_dataset/src \
  vietlawbert-app \
  python run_crawl_shards.py \
    --total-documents 160660 \
    --page-size 100 \
    --pages-per-shard 10 \
    --concurrency 16 \
    --download-delay 0 \
    --allow-upstream-missing \
    --defer-ocr \
    --output-dir ../artifacts/full_crawl_v5
```

Không truyền thẳng `.jsonl.gz` cho `scrapy -O`: Scrapy 2.17 hiểu `gz` là output format và fail.
Runner xử lý đúng chuỗi raw → gate → gzip → gate. Sau tất cả shard, runner kiểm tra duplicate ID
xuyên shard và ghi `full_crawl_gate.json`; chỉ `passed=true` mới được chuyển sang ingestion.

Full crawl dùng hai pha nội dung:

- `hasContent=false` và file fingerprint `Template.pdf` 32.052 byte được ghi
  `UPSTREAM_TEMPLATE`, không tải/OCR file trắng lặp lại.
- PDF scan có digital text dưới ngưỡng được ghi `OCR_PENDING` khi dùng `--defer-ocr`.
- Cả hai nhóm nằm trong `*.quarantine.jsonl`; chỉ hai trạng thái có bằng chứng này được phép qua
  raw gate. Mọi HTML rỗng không giải thích được vẫn làm shard fail.
- Sau raw crawl, gom ID `OCR_PENDING` và chạy targeted retry trong container với
  `OCR_INLINE_ENABLED=1`. Runner chuẩn tự batch, overlay và audit:

```bash
docker run --rm \
  -v /mnt/d/vietlawbert:/app \
  -w /app/law_dataset/src \
  -e OCR_CONCURRENCY=1 \
  vietlawbert-app \
  python run_ocr_quarantine.py \
    --input-dir ../artifacts/full_crawl_v5 \
    --batch-size 2
```

Runner bảo toàn base, tạo `*.rescued.jsonl.gz`, `ocr_retries/ocr_state.json` và file unresolved.
Lượt incremental tự bỏ qua shard đã xử lý. Smoke thật trên shard 4 và 6 phục hồi 8/8 scan,
0 unresolved; hai artifact rescued đều pass gate.

Không OCR inline trong full network crawl: phép đo shard 4 cho thấy callback OCR đồng bộ từng làm
throughput rơi từ 144 xuống 2 item/phút. Sau khi tách pha, throughput phút ổn định đạt
157–214 item/phút. Targeted OCR dùng `OCR_CONCURRENCY=2` để không bão hòa CPU.

Trước khi mở full embedding:

1. Benchmark GPU/remote provider trên đúng pilot 2.125 chunk và kiểm tra chất lượng retrieval.
2. Chọn collection/model version duy nhất; không trộn INT8, FP32 và provider khác nhau.
3. Cấu hình retention/dung lượng Redpanda và giám sát lag, disk, 403/429.
4. Chuyển mọi record thiếu text sang rescue PDF/OCR; không commit record rỗng.
5. Audit từng checkpoint về ID, HTML, ontology, Milvus–Neo4j và chunk size.

Raw artifact 1.000 record là 67,53 MiB; ước lượng full raw corpus 10,59 GiB. Trước khi chạy,
duy trì tối thiểu 20 GiB trống cho artifact, retry, log và overhead. Tại lần kiểm tra gần nhất ổ D
còn 43,73 GiB. Với profile đã đo, raw crawl dự kiến khoảng 13,7 giờ; nếu có 403/429, hạ
`CRAWLER_CONCURRENCY` từ 16 xuống 8 và giữ nguyên các shard đã pass.

`audit_pilot.py` xử lý JSONL theo streaming nên không nạp artifact 10+ GiB vào RAM. Runner đã được
smoke-test cả lượt crawl mới và lượt resume `SKIP PASS`.

Với CPU hiện tại, ngoại suy full embedding khoảng 18 ngày liên tục; đây là lý do kỹ thuật để
không phát lệnh full embedding trước khi có GPU/service phù hợp.
