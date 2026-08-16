# Hướng dẫn vận hành VietLawBERT: từ Ubuntu đến full crawl

*Cập nhật: 16/08/2026. Đây là runbook chạy toàn bộ corpus quan sát được (160.660 record),
đưa raw document qua Kafka và archive manifest vào Neo4j. Không chunk, embedding, train hay xây mô hình.*

## 1. Mở Ubuntu và vào project

Từ PowerShell trên Windows:

```powershell
wsl -d Ubuntu-24.04
```

Trong Ubuntu:

```bash
cd /mnt/d/vietlawbert
docker version
docker compose config --quiet
```

Nếu `docker version` không kết nối được daemon, bật Docker Desktop và WSL Integration cho
Ubuntu-24.04 trước khi tiếp tục.

## 2. Build image và bật hạ tầng dữ liệu

```bash
cd /mnt/d/vietlawbert
docker compose build app
docker compose up -d redpanda neo4j etcd minio milvus-standalone
docker compose ps
```

Image triển khai dùng PyTorch CPU, không kéo CUDA/NVIDIA; có Playwright Chromium và Tesseract
`vie+eng`. Kiểm tra:

```bash
docker run --rm vietlawbert-app bash -lc \
  'python -c "import torch; print(torch.__version__, torch.version.cuda)"; tesseract --list-langs'
```

## 3. Preflight

```bash
docker run --rm \
  -v /mnt/d/vietlawbert:/app \
  -w /app \
  vietlawbert-app \
  python -m pytest tests -q
```

Chỉ chạy crawl khi test pass và các service `redpanda`, `neo4j`, `milvus-standalone` đang chạy.

## 4. Điều kiện đã đạt trước full crawl

Pilot 8.000 đã pass: 8.000 ID duy nhất, OCR phục hồi 13/13 scan, Kafka lag 0 và ID/SHA-256
artifact–Kafka–Neo4j khớp hoàn toàn. Không cần crawl lại pilot. Full run phải dùng thư mục,
topic, consumer group và `dataset_id` mới như bên dưới.

Nguồn tại thời điểm kiểm định báo 160.660 record. Trước khi chạy cần giữ ít nhất 20 GiB trống:

```bash
df -h /mnt/d
docker system df
```

Nếu tổng record trên nguồn thay đổi, thay đồng bộ `160660` ở tất cả lệnh và tính lại số shard
theo `ceil(total/1000)`. Với snapshot hiện tại, số shard là 161.

## 5. Crawl toàn bộ 160.660 văn bản

Raw crawl không phát Kafka và không OCR inline. Runner crawl từng shard 1.000 record, audit file
thô, nén gzip, audit lại và ghi checkpoint. Shard cuối chứa 660 record.

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
    --output-dir ../artifacts/full_crawl_v6
```

Không đóng terminal đang chạy job. Từ terminal Ubuntu thứ hai, theo dõi checkpoint:

```bash
cd /mnt/d/vietlawbert
watch -n 30 'jq "{complete, completed: (.completed_shards|length), total_documents}" \
  law_dataset/artifacts/full_crawl_v6/crawl_state.json'
```

Xem log shard mới nhất:

```bash
tail -f "$(ls -1t law_dataset/artifacts/full_crawl_v6/*.jsonl.log | head -1)"
```

Nếu job hoặc máy dừng, chạy lại **nguyên lệnh crawl**. Runner sẽ audit và in `SKIP PASS` cho shard
đã hoàn tất; không crawl lại checkpoint hợp lệ. Không xóa cả thư mục để resume.

Theo dõi HTTP 403/429 trong log. Nếu nguồn bắt đầu throttling, dừng có kiểm soát bằng `Ctrl+C`,
đổi `--concurrency 16` thành `8` rồi chạy lại nguyên lệnh.

## 6. Kiểm tra gate raw sau khi crawl xong

```bash
jq . law_dataset/artifacts/full_crawl_v6/full_crawl_gate.json
```

Chỉ đi tiếp khi có:

```text
expected_documents       = 160660
unique_documents         = 160660
cross_shard_duplicates   = 0
passed                   = true
```

`content_complete=false` tại đây chưa nhất thiết là lỗi: raw crawl cho phép các PDF scan được
đánh dấu `OCR_PENDING` và placeholder chính thức `UPSTREAM_TEMPLATE`. Các lỗi rỗng không giải
thích được vẫn làm shard fail.

## 7. OCR toàn bộ quarantine ở worker riêng

```bash
docker run --rm \
  -v /mnt/d/vietlawbert:/app \
  -w /app/law_dataset/src \
  -e OCR_CONCURRENCY=1 \
  vietlawbert-app \
  python run_ocr_quarantine.py \
    --input-dir ../artifacts/full_crawl_v6 \
    --batch-size 2
```

Worker chỉ OCR các record `OCR_PENDING`, giữ nguyên base và tạo `*.rescued.jsonl.gz`. Nếu bị
dừng, chạy lại nguyên lệnh; state nằm tại:

```bash
jq . law_dataset/artifacts/full_crawl_v6/ocr_retries/ocr_state.json
```

Phải review mọi entry có `unresolved > 0`; không đổi tài liệu không đọc được thành `VALID` giả.
Canonical selector ở các bước sau tự ưu tiên file `.rescued.jsonl.gz` nếu tồn tại.

## 8. Phát toàn bộ canonical corpus lên Kafka

Dùng topic mới hoàn toàn; publisher từ chối topic đã có message để tránh đếm trùng:

```bash
docker run --rm --network vietlawbert_vietlaw_net \
  -v /mnt/d/vietlawbert:/app \
  -w /app/law_dataset/src \
  -e KAFKA_BROKER=redpanda:29092 \
  vietlawbert-app \
  python publish_crawl_artifacts.py \
    --input-dir ../artifacts/full_crawl_v6 \
    --expect-documents 160660 \
    --expect-shards 161 \
    --topic law-documents-v7-full-gzip \
    --partitions 24
```

Mỗi JSON đầy đủ, gồm `html_raw`, được nén gzip và kèm SHA-256 trước khi phát. Chỉ đi tiếp nếu
publisher in `delivered: 160660`.

## 9. Chạy raw archive consumer cho toàn bộ corpus

```bash
docker run --rm --network vietlawbert_vietlaw_net \
  -v /mnt/d/vietlawbert:/app \
  -w /app/law_dataset/src \
  -e KAFKA_BROKER=redpanda:29092 \
  -e KAFKA_RAW_GROUP_ID=raw-archive-v7-full-gzip \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=vietlawbert \
  vietlawbert-app \
  python raw_archive_consumer.py \
    --topic law-documents-v7-full-gzip \
    --dataset-id v7-full-gzip \
    --expect-documents 160660 \
    --batch-size 500 \
    --idle-exit-seconds 10
```

Consumer này giải nén, kiểm tra payload rồi `MERGE` manifest vào Neo4j theo
`(dataset_id, doc_id)`. Nó commit Kafka offset chỉ sau khi batch DB thành công, có thể replay an
toàn và **không load BGE-M3, không chunk, không embedding**.

## 10. Chứng minh full corpus khớp artifact–Kafka–Neo4j

Kiểm tra nhanh:

```bash
docker exec neo4j_vietlaw cypher-shell -u neo4j -p vietlawbert \
  'MATCH (d:RawLawDocument {dataset_id:"v7-full-gzip"}) RETURN count(d) AS documents, count(DISTINCT d.doc_id) AS unique_documents;'

docker exec neo4j_vietlaw cypher-shell -u neo4j -p vietlawbert \
  'MATCH (d:RawLawDocument {dataset_id:"v7-full-gzip"}) RETURN d.html_status, count(*) ORDER BY d.html_status;'

docker exec redpanda_vietlaw rpk group describe raw-archive-v7-full-gzip
```

Kết quả bắt buộc: `documents=160660`, `unique_documents=160660` và Kafka `TOTAL-LAG=0`.

Chạy gate mạnh để so toàn bộ tập ID và SHA-256:

```bash
docker run --rm --network vietlawbert_vietlaw_net \
  -v /mnt/d/vietlawbert:/app \
  -w /app/law_dataset/src \
  -e KAFKA_BROKER=redpanda:29092 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=vietlawbert \
  vietlawbert-app \
  python verify_pipeline.py \
    --input-dir ../artifacts/full_crawl_v6 \
    --expect-shards 161 \
    --expect-documents 160660 \
    --topic law-documents-v7-full-gzip \
    --dataset-id v7-full-gzip \
    --output ../artifacts/full_crawl_v6/pipeline_full_gate.json
```

Chỉ kết luận full pipeline đạt khi `passed=true` và cả bốn trường
`artifact_kafka_id_match`, `artifact_database_id_match`, `artifact_kafka_hash_match`,
`artifact_database_hash_match` đều `true`.

## 11. Thời gian dự kiến và phạm vi của con số 18 giờ

Pilot đo được 8.000 record trong 47,82 phút, tương đương 167,30 record/phút. Ngoại suy 160.660
record là khoảng 16,0 giờ raw crawl. Nên dành cửa sổ 16–20 giờ vì tốc độ nguồn, retry và
throttling có thể thay đổi.

Raw archive consumer đã đạt 376,176 document/giây trên pilot; nếu giữ được tốc độ này, 160.660
document mất khoảng 7,1 phút, nên dành 10–20 phút cho consumer và commit. Publish Kafka và gate
hash cần thêm thời gian vận hành. OCR phụ thuộc số trang scan và không có SLA chắc chắn; nó có
thể làm tổng thời gian vượt 18 giờ.

Vì vậy, **18 giờ là mốc hợp lý nhưng không phải cam kết** cho crawl + raw archive consumer trong
điều kiện nguồn ổn định. Kế hoạch an toàn cho toàn bộ pipeline không embedding là 18–22 giờ, cộng
thời gian review OCR unresolved nếu có.

Con số này tuyệt đối không bao gồm consumer chunk + BGE-M3 embedding. Với CPU đã đo khoảng
2,2 chunk/giây, full embedding từng được ngoại suy khoảng 18 ngày; pha đó chỉ nên chạy sau bằng
GPU hoặc embedding endpoint và không thuộc runbook này.
