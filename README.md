# VietLawBERT

**Hierarchy-Aware Matryoshka Representation Learning & Heterogeneous Graph-Guided Hybrid Retrieval cho Văn bản Quy phạm Pháp luật Việt Nam.**

Dự án nghiên cứu xây dựng mô hình nhúng ngữ nghĩa chuyên biệt cho VBQPPL Việt Nam và một kiến trúc truy xuất lai (RAG) có căn cứ kiểm chứng, nối tiếp bài **VNLawBERT** (Chau et al., NICS 2020 — nền BERT-Base/mBERT, trọng số không công khai).

> Câu hỏi nghiên cứu RQ1–RQ4 xem đề cương `docs/DeCuongTN-v3.1-filan.pdf`. Kế hoạch chạy đầu-cuối và trạng thái xem `docs/QuyTrinh-RunCloud-TongThe.md` + `docs/WeeklyReport-*.md`.

---

## 1. Kiến trúc 5 giai đoạn

```
Pha 1          Pha 2             Pha 3          Pha 4                    Pha 5
CRAWL     →    PARSE & INGEST →  ĐỒ THỊ HIN  →  HUẤN LUYỆN MRL      →    TRUY XUẤT + ĐÁNH GIÁ
(vbpl.vn)      Qdrant/ES/Mongo   Neo4j          7 backbone + biến thể    Hybrid RAG + RQ1–RQ4
```

| Giai đoạn | Module | Vai trò |
| --- | --- | --- |
| 1. Thu thập | `crawler/` (Scrapy + Playwright, `shard_runner`) | Cào VBQPPL phân mảnh, khử trùng theo `doc_number`, checkpoint resume |
| 2. Tiền xử lý | `preprocess/` (AST parser Lark, text cleaner), `quality/` | Bóc tách Phần→Chương→Mục→Điều→Khoản→Điểm, định danh 4 tầng (doc/chương/điều/khoản) |
| 2. Nạp đa kho | `pipeline/ingest_pipeline`, `database/` | Qdrant (dense bge-m3), Elasticsearch (BM25), Neo4j (HIN), MongoDB (toàn văn + metadata) |
| 3. Đồ thị HIN | `database/build_hin_graph` | 22 loại quan hệ (THAY_THE, SUA_DOI_BO_SUNG, DAN_CHIEU, BAI_BO…); positive/negative đúng cực tính cho khai phá mẫu âm |
| 4. Huấn luyện | `training/` (`generate_hin_triplets`, `train_mrl`) | HIN-Guided Hard Negatives + Hierarchy-Aware Matryoshka InfoNCE, quét 6 lát cắt d {64…1024} |
| 5. Truy xuất & đánh giá | `rag/retriever`, `benchmark/` (`evaluate_rqs`, `evaluate_rrf`, `vlqa_eval`, `build_multihop_subset`, `llm_judge`) | RRF (k=60) + graph rerank; đánh giá trên VLQA + subset multi-hop |
| Điều phối | `main.py`, `scripts/`, `configs/` | CLI theo pha + script một mạch + deploy cloud |

**Bộ đánh giá:** VLQA (VLSP DRiLL 2025 — benchmark ngoài, chỉ để đánh giá) + **subset multi-hop tự sinh** từ đồ thị HIN (đo multi-hop/completeness mà VLQA không gánh được).

**Bốn câu hỏi nghiên cứu:**
- **RQ1** — Hiệu quả chiến lược sinh mẫu âm: so `random` / `BM25 lexical` / `HIN metapath` (cùng dữ liệu, trên bge-m3).
- **RQ2** — Năng lực Hierarchy-Aware MRL: Pareto RAM–Hit@10 qua 6 lát cắt d + Silhouette tại d=64 so với MRL phẳng (FlatMRL).
- **RQ3** — Bóc tách kiến trúc truy xuất lai: 4 chế độ `sparse_only` / `dense_only` / `hybrid` / `hybrid + graph rerank`, kèm độ trễ P95.
- **RQ4** — Chất lượng sinh & khử ảo giác: LLM-as-judge.

Metric tất định (Hit/MRR/NDCG/MAP@k + set_recall + completeness), lưu điểm **per-query**, kiểm định **paired t-test + Wilcoxon**.

---

## 2. Cài đặt

```bash
# 1. Môi trường Python
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium        # crawl dùng scrapy-playwright → cần browser

# 2. Hạ tầng CSDL (Docker)
docker compose up -d qdrant elasticsearch neo4j db redis    # 'db' = MongoDB

# 3. Cấu hình
cp .env.example .env     # rồi điền khoá + đường dẫn (xem mục 4)
```

**GPU:** chạy được trên CPU (mặc định an toàn) nhưng **huấn luyện và nhúng corpus cần GPU**. GPU cũ (vd GTX 1050 Ti, CC 6.1) không tương thích `torch` mặc định → tự fallback CPU → không train/embed được; thí nghiệm quy mô lớn thuê Cloud GPU (mục 3.2).

---

## 3. Chạy

### 3.1. Theo từng pha (local)

```bash
python main.py --phase check        # kiểm tra hạ tầng 4 CSDL
python main.py --phase crawl        # Pha 1 (cào, khử trùng doc_number) — chỉ cần mạng, không GPU
python main.py --phase ingest       # Pha 2 (nhúng → Qdrant + ES) — cần GPU
python main.py --phase graph        # Pha 3 (dựng Neo4j HIN)
python main.py --phase build-data   # Pha 1→3 gộp (crawl→ingest→graph, WIPE dựng lại từ đầu)
python main.py --phase train        # Pha 4 (train MRL một backbone)
python main.py --phase benchmark    # Pha 5 (đánh giá VLQA + RQ)
python main.py --ask "Vượt đèn đỏ xe máy phạt bao nhiêu?" --top-k 3   # hỏi đáp RAG
```

### 3.2. Một mạch A→B (trên máy có GPU)

`scripts/run_all_experiments.sh` chạy liền: crawl → ingest → graph → 2 cổng kiểm tra → nạp VLQA → sinh multi-hop → re-mine triplet → train 7 backbone + 3 biến thể bge-m3 (RQ1/RQ2) → bảng đối chứng (7 + mBERT zero-shot + 3 biến thể) → RQ1/RQ3/RQ4 + eval VLQA.

```bash
BUILD_DATA=1 ./scripts/run_all_experiments.sh cuda      # dựng lại data từ đầu rồi chạy hết
BUILD_DATA=0 ./scripts/run_all_experiments.sh cuda      # bỏ qua pha dữ liệu, dùng graph/triplet sẵn có
```

### 3.3. Deploy lên Cloud GPU (chạy từ máy local)

```bash
bash scripts/auto_cloud_runner.sh <IP> <PORT> <USER> <PASS>
```

Script tự: đồng bộ code (rsync, loại file nặng) → vá `.env` box (`DATA_STORAGE_ROOT`, `VLQA_DIR`, `EMBED_DEVICE=cuda`) → tải VLQA từ Drive (`scripts/fetch_vlqa.sh`) → cài `requirements.txt` + Playwright → chạy `run_all_experiments.sh` trong tmux. Dữ liệu được crawl lại trên box; chỉ VLQA (nguồn ngoài repo, ~120 MB) tải từ Drive.

---

## 4. Cấu hình `.env`

| Nhóm biến | Ý nghĩa |
| --- | --- |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Kết nối Neo4j |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION_NAME` | Qdrant (dense) |
| `ES_HOST` / `ES_INDEX_NAME` | Elasticsearch (BM25) |
| `MONGO_URI` / `MONGO_DB_NAME` | MongoDB (toàn văn) |
| `DATA_STORAGE_ROOT` | Thư mục dữ liệu thô (shard, log). Trỏ sai/không tồn tại → tự fallback `/mnt/data` |
| `EMBED_DEVICE` | `cpu` hoặc `cuda` |
| `BASE_MODEL_NAME` | Backbone nhúng corpus (`BAAI/bge-m3`) |
| `VLQA_DIR` | Thư mục VLQA (`train.json` + `legal_corpus.json`) |
| `EVAL_API_KEY` / `EVAL_API_BASE` / `EVAL_MODEL` | **LLM-as-judge (RQ4) + sinh multi-hop + chấm correctness VLQA** (endpoint OpenAI-compatible) |
| `PRIMARY_/FALLBACK_/ROUTER_LLM_*` | Cascade LLM tầng sinh (hỏi đáp RAG) |

> `.env` **bị gitignore** (chứa mật khẩu DB + nhiều API key) → **không bao giờ đẩy lên GitHub**. Lên box cloud thì scp/rsync trực tiếp (auto_cloud_runner lo việc này), không qua git.

---

## 5. Kiểm thử

```bash
docker compose up -d qdrant elasticsearch neo4j db redis   # bật service cho test cần CSDL
python -m pytest tests -q                                   # 256 test (test thiếu service/GPU tự SKIP)
python -m pytest tests/unit -q                              # unit thuần (metric, triplet, multihop, remap...)
python -m pytest tests/system -q                            # hệ thống 5 tầng
python -m pytest tests/model -q                             # mô hình (MRL loss, khai phá mẫu âm)
```

Thực nghiệm RQ qua **promptfoo** chạy từ `tests/*/configs/` (vd `promptfooconfig.rq3_retrieval.yaml`, `promptfooconfig.generation.yaml`).

---

## 6. Cấu trúc thư mục

```text
vietlawbert/
├── src/
│   ├── configs/            # config.py, paths.py, logging_config.py, global_id.py
│   ├── crawler/            # Pha 1: Scrapy spider + shard runner + OCR
│   ├── preprocess/         # Pha 2a: AST parser, text cleaner, html→md
│   ├── quality/            # Pha 2b: kiểm định corpus + kiểm toán crawl
│   ├── pipeline/           # Pha 2c: ingest đa kho
│   ├── database/           # Pha 3: Neo4j HIN, Qdrant/Mongo client
│   ├── training/           # Pha 4: HIN triplets (--neg-strategy), train MRL
│   ├── rag/                # Pha 5a: hybrid retriever, es retriever, generator
│   ├── benchmark/          # Pha 5b: evaluate_rqs, evaluate_rrf, vlqa_eval, build_multihop_subset, llm_judge
│   ├── mcp_servers/        # giao thức MCP cho AI Agents
│   ├── demo/               # giao diện web Gradio
│   └── cli/                # wrapper lệnh con
├── tests/                  # unit / system / model / evaluation / _common
├── scripts/                # run_all_experiments.sh, auto_cloud_runner.sh, fetch_vlqa.sh, run_p0_to_p4.sh
├── outputs/                # runtime: artifacts (triplets), benchmark (VLQA + .csv/.tex), checkpoints, models, logs
├── docs/                   # đề cương, tổng quan, quy trình, báo cáo tuần
├── main.py                 # CLI điều phối theo pha
├── docker-compose.yml      # Qdrant, Elasticsearch, Neo4j, MongoDB, Redis
└── requirements.txt
```

---

## 7. Trạng thái & hạn chế đã biết

- **Chạy lại từ đầu:** corpus được cào mới từ vbpl.vn, khử trùng theo `doc_number`. 4 CSDL dựng lại trên box cloud.
- **Train/embed cần Cloud GPU** — máy local (GTX 1050 Ti) chỉ validate được pha crawl; pha nhúng/huấn luyện chạy trên A100.
- **RQ4 + sinh multi-hop cần `EVAL_API_KEY`** (hhtechapi/OpenAI-compatible). Thiếu key thì bước multi-hop và judge bị bỏ qua.
- **Trục 2 (thích ứng thời gian):** hàm `remap_gold` (THAY_THE/BAI_BO = thay, SUA_DOI = hợp) đã có + test; cắm vào vòng chấm là bước kế trên box.
- **Chất lượng mô hình** chỉ nghiệm thu được sau lượt chạy A100 đầu tiên; khuyến nghị smoke `--total-docs` nhỏ trước khi cào full.

## 8. Giấy phép dữ liệu

Dữ liệu VBQPPL là tài sản công (Nghị định 47/2020/NĐ-CP), thu thập phục vụ nghiên cứu học thuật phi thương mại từ Cổng thông tin pháp luật Quốc gia (vbpl.vn). VLQA trích dẫn theo biên bản VLSP DRiLL 2025.
