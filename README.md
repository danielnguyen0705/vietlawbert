# VietLawBERT

**Hierarchy-Aware Matryoshka Representation Learning & Heterogeneous Graph-Guided Hybrid Retrieval for Vietnamese Legal Texts.**

Hệ thống nghiên cứu xây dựng mô hình nhúng ngữ nghĩa chuyên biệt cho Văn bản Quy phạm Pháp luật (VBQPPL) Việt Nam và một kiến trúc truy xuất lai (RAG) có căn cứ kiểm chứng, phát triển nối tiếp bài báo nền tảng **VNLawBERT (2020)**.

> Mục tiêu khoa học và các câu hỏi nghiên cứu (RQ1–RQ4) xem đề cương `DeCuongTN-v3.1.docx`. Trạng thái tiến độ & hạn chế hiện tại xem "Ma trận đặc tả tiến độ VietLawBERT (v3.1)".

---

## 1. Kiến trúc 5 giai đoạn

```
Pha 1          Pha 2             Pha 3            Pha 4                 Pha 5
CRAWL     →    AST PARSE &   →   ĐỒ THỊ HIN   →   HUẤN LUYỆN       →    TRUY XUẤT + SINH
(vbpl.vn)      INGEST            Neo4j + Graph    VietLawBERT-MRL       Hybrid RAG + Guardrail
               (Qdrant/ES)       Embedding 128d   (HIN hard negatives)  (RRF + Graph rerank)
```

| Giai đoạn | Module nguồn | Vai trò |
| --- | --- | --- |
| 1. Thu thập | `crawler/` (Scrapy + Playwright, `shard_runner`, `ocr_runner`) | Cào VBQPPL phân mảnh, ≤5 req/s, checkpoint resume |
| 2. Tiền xử lý | `preprocess/` (`ast_parser` Lark, `text_cleaner`, `html_to_md`), `quality/` (bộ lọc chất lượng) | Bóc tách cây cú pháp Phần→Chương→Mục→Điều→Khoản→Điểm, nhúng metadata |
| 2. Nạp đa kho | `pipeline/` (`ingest_pipeline`, `ast_batch_processor`, `neo4j_batch_loader`), `database/` (`qdrant_client`, `neo4j_client`, `build_hin_graph`) | Qdrant (dense 256d), Elasticsearch (BM25), Neo4j (HIN), MongoDB |
| 3. Đồ thị & embedding | `database/build_hin_graph`, `training/compute_graph_embeddings` | HIN 22 quan hệ hiệu lực/dẫn chiếu + node2vec/RotatE 128d |
| 4. Khai phá & huấn luyện | `training/` (`generate_hin_triplets`, `train_mrl`) | HIN-Guided Hard Negatives (metapath + RWR, β=0.5) + Hierarchy-Aware Matryoshka InfoNCE |
| 5. Truy xuất & sinh | `rag/` (`retriever` hybrid, `es_retriever`, `generator` cascade) | RRF (k=60) + Graph rerank + LLM sinh có trích dẫn |
| Đánh giá | `benchmark/` (`build_vietlawbench`, `evaluate_rrf`, `evaluate_rqs`, `llm_judge`) | VietLawBench 1000 câu, RQ1–RQ4, paired t-test |
| Điều phối | `main.py`, `cli/`, `configs/` | CLI hợp nhất + cấu hình (hub `configs/`) |

---

## 2. Yêu cầu & cài đặt

```bash
# 1. Môi trường Python
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

# 2. Hạ tầng CSDL (Docker)
docker compose up -d qdrant elasticsearch neo4j db redis

# 3. Cấu hình
cp .env.example .env     # rồi điền khoá (xem mục 4)
```

GPU: dự án chạy được trên CPU (mặc định an toàn). Nếu dùng GPU cũ (vd GTX 1050 Ti, CC 6.1) không tương thích bản `torch` mặc định, cài bản khớp: `pip install torch --index-url https://download.pytorch.org/whl/cu126`. Thí nghiệm quy mô lớn nên thuê Cloud GPU (xem `auto_cloud_runner.sh`, `cloud_entrypoint.py`).

---

## 3. Chạy pipeline

```bash
python main.py                 # menu tương tác (0–9)

# hoặc theo từng pha:
python main.py --phase check       # kiểm tra hạ tầng + health check
python main.py --phase crawl       # Pha 1
python main.py --phase ingest      # Pha 2 (AST + Qdrant + ES)
python main.py --phase graph       # Pha 3 (HIN + graph embedding)
python main.py --phase train       # Pha 4 (VietLawBERT-MRL)
python main.py --phase benchmark   # Pha 5 (Master baseline + RQ1–RQ4)
python main.py --phase post-ingest # Pha 3 → 5
python main.py --phase all         # toàn trình Pha 1 → 5

# Hỏi đáp RAG trực tiếp:
python main.py --ask "Đi xe máy trên vỉa hè bị phạt bao nhiêu?" --top-k 3
```

Chạy toàn bộ thực nghiệm (có thuê GPU): `bash run_all_experiments.sh`.

---

## 4. Cấu hình `.env` (các biến chính)

| Biến | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Kết nối Neo4j | `bolt://localhost:7687` / `neo4j` / `vietlawbert2026` |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION_NAME` | Qdrant | `localhost` / `6333` / `vietlawbert_chunks` |
| `ES_HOST` / `ES_INDEX_NAME` | Elasticsearch | `http://localhost:9200` / `vietlaw_sparse_idx` |
| `BASE_MODEL_NAME` | Backbone encoder | `BAAI/bge-m3` |
| **Cascade LLM (tầng sinh)** | | |
| `PRIMARY_LLM_MODEL` / `_API_BASE` / `_API_KEY` | Tầng 1 SOTA | `gpt-4o` / `https://hhtechapi.net/v1` / **(cần điền)** |
| `FALLBACK_LLM_MODEL` / `_API_BASE` / `_API_KEY` | Tầng 2 nhanh | `gpt-6-sol` / `https://hhtechapi.net/v1` / **(cần điền)** |
| `GENERATOR_MODEL` / `LLM_API_BASE` / `LLM_API_KEY` | Tầng 3 local | `qwen2.5:7b-instruct` / `http://localhost:11434/v1` / `ollama` |
| `EVAL_MODEL` / `EVAL_API_BASE` | Judge RAGAS (RQ4) | `gpt-6-sol` / `https://hhtechapi.net/v1` |

> ⚠️ Cascade **bỏ qua tầng cloud nếu API key rỗng**. Để RQ4/hỏi đáp dùng được cloud: điền khoá hhtechapi vào `PRIMARY_LLM_API_KEY` và `FALLBACK_LLM_API_KEY` (judge tự dùng lại khoá Primary). Tầng local: `ollama pull qwen2.5:7b-instruct`.

---

## 5. Kiểm thử

Hai framework: **pytest** (toàn vẹn 5 tầng) + **promptfoo** (thực nghiệm RQ), tổ chức thành `tests/system`, `tests/model` và `tests/_common` (dùng chung). Chi tiết: `tests/README.md`.

```bash
docker compose up -d qdrant elasticsearch neo4j     # bật service cho test requires_*
python -m pytest tests -v                           # toàn bộ (test thiếu service sẽ tự SKIP)
python -m pytest tests/system -v                    # chỉ nhánh hệ thống RAG (dữ liệu → đồ thị → truy xuất → sinh → chatbox)
python -m pytest tests/model -v                     # chỉ nhánh mô hình (MRL loss, khai phá mẫu âm)
python tests/run_audit.py                           # chạy tuần tự + báo cáo

# Thực nghiệm RQ (promptfoo) — mỗi nhánh chạy từ configs/ của nhánh đó:
cd tests/model/configs
promptfoo eval -c promptfooconfig.baselines.yaml -o ../out/eval-baselines.json      # baseline đa backbone
cd ../../system/configs
promptfoo eval -c promptfooconfig.rq3_retrieval.yaml -o ../out/eval-rq3_retrieval.json  # RQ3 (ablation truy xuất)
promptfoo eval -c promptfooconfig.generation.yaml -o ../out/eval-generation.json        # RQ4 (faithfulness)
```

---

## 6. Cấu trúc thư mục

```
vietlawbert/
├── main.py                 # CLI điều phối toàn trình
├── cli/                    # wrapper lệnh con (crawl, ingest, ocr, audit, verify, ...)
├── configs/                # config.py, paths.py, logging_config.py (hub cấu hình)
├── crawler/                # Scrapy spider + shard runner + OCR
├── preprocess/             # AST parser, text cleaner, html→md
├── quality/                # bộ lọc chất lượng + kiểm toán crawl/pipeline
├── pipeline/               # ingest, ast batch, neo4j batch loader
├── database/               # neo4j/qdrant client, build HIN graph, cypher templates
├── training/               # HIN triplets, train MRL, graph embeddings
├── rag/                    # hybrid retriever, es retriever, generator (cascade)
├── benchmark/              # build VietLawBench, evaluate_rrf, evaluate_rqs, llm_judge
├── tests/                  # system/ · model/ · _common/ (pytest + promptfoo; xem tests/README.md)
├── docker-compose.yml      # Qdrant, Elasticsearch, Neo4j+APOC, MongoDB, Redis
└── requirements.txt
```

---

## 7. Trạng thái & hạn chế đã biết

- **GPU cũ không tương thích** `torch` mặc định → chạy CPU → chậm (cold-start cache đồ thị ~27s). Cài `torch` bản `cu126` hoặc thuê Cloud GPU cho thí nghiệm lớn.
- **RQ4 cần khoá LLM**: điền khoá hhtechapi + `ollama pull` (mục 4). Xác nhận id model với bảng giá hhtechapi (`gpt-6-sol` đã xác nhận chạy).
- **Chất lượng mô hình MRL**: trên một số backbone mạnh bản tinh chỉnh chưa vượt zero-shot; `phobert-base-v2` là backbone cải thiện tốt nhất hiện tại. Chi tiết & hướng xử lý: xem ma trận tiến độ.

## 8. Giấy phép dữ liệu
Dữ liệu VBQPPL là tài sản công (Nghị định 47/2020/NĐ-CP), thu thập phục vụ nghiên cứu học thuật phi thương mại từ Cổng thông tin pháp luật Quốc gia (vbpl.vn).
