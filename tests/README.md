# Kịch bản kiểm thử VietLawBERT (bám đặc tả tiến độ v3.1)

Bộ test được tổ chức theo **5 tầng kỹ thuật** (ánh xạ 1-1 với 4 giai đoạn trong đề cương) cộng
một thư mục **thực nghiệm 4 RQ** chạy bằng promptfoo. Hai framework sử dụng:

- **pytest** — kiểm định *tính đúng & bất biến* (unit + integration + invariant). Chạy nhanh, tất định, dùng cho CI.
- **promptfoo** — chạy *thực nghiệm khoa học* (benchmark 1000 câu, ablation, LLM-as-a-Judge) cho RQ1–RQ4.

> Đề xuất bổ sung (không bắt buộc): khi lên Cloud GPU, thêm `pytest-benchmark` để chốt P50/P95/P99 độ trễ,
> và `ragas` (thư viện gốc) cho RQ4 thay vì chỉ llm-rubric — xem mục Framework bên dưới.

## Tổng quan: 28 file (17 file test pytest + 5 file hạ tầng + 6 file harness promptfoo)

```
tests/
├── conftest.py              [MỚI] nối sys.path portable + fixture neo4j/qdrant/es (tự SKIP khi service tắt)
├── pytest.ini               [MỚI] markers tier1..5 + requires_* + testpaths
├── run_audit.py             [KẾ THỪA] chạy tuần tự cả 5 tầng, in báo cáo tổng
├── _common/
│   ├── __init__.py          [MỚI]
│   └── paths.py             [MỚI] dò gốc repo + đọc env VIETLAWBERT_ROOT/DATA/CKPT (hết hard-code)
│
├── tier1_data_curation/            # Giai đoạn 1 — Dữ liệu & AST
│   ├── test_tier1_data_curation.py [KẾ THỪA] 12 ca: gate ngôn ngữ, metadata regex, AST đa cấp, corpus parquet/gzip
│   ├── test_ast_parser.py          [DỜI] unit AST hierarchy tracker + văn bản có cấu trúc
│   ├── test_legal_chunker.py       [DỜI] unit cắt chunk theo đơn vị pháp lý
│   ├── test_text_cleaner.py        [DỜI] unit khử boilerplate + trích số hiệu/ngày
│   ├── test_crawl_audit.py         [DỜI] unit kiểm toán crawl
│   └── test_quality_text_gate.py   [MỚI] ĐỊNH LƯỢNG ngưỡng nguyên âm 0.05 (loại rác, không loại nhầm)
│
├── tier2_graph_multistore/         # Giai đoạn 1 — Đồ thị HIN & đa kho
│   ├── test_tier2_graph_multistore.py  [KẾ THỪA] ràng buộc Neo4j, chunk mồ côi, self-loop, Qdrant status
│   ├── test_hin_22_relations.py        [MỚI] đếm phổ quan hệ, khẳng định đủ quan hệ hiệu lực trọng yếu
│   ├── test_metadata_injection.py      [MỚI] [META]/[HIERARCHY]/[CONTENT] nhúng đúng + số hiệu VB
│   ├── test_ingestion_stabilization.py [DỜI] integration ổn định nạp dữ liệu
│   └── test_merge_crawl_artifacts.py   [DỜI] integration gộp artifact crawl
│
├── tier3_representation_mrl/       # Giai đoạn 2 — Biểu diễn MRL
│   ├── test_tier3_representation_mrl.py [KẾ THỪA] toán InfoNCE 6 lát cắt, L2 sau cắt, gradient, Spearman
│   ├── test_mrl_loss.py                [DỜI] unit forward/backward MRL loss
│   ├── test_hierarchy_loss.py          [MỚI] cô lập gamma*L_Hierarchy (SupCon d=64) có tham gia loss
│   └── test_hin_triplet_miner.py       [MỚI] khẳng định entrypoint khai phá mẫu âm (training.generate_hin_triplets, beta=0.5)
│
├── tier4_retrieval_ir/             # Giai đoạn 3 — Truy xuất
│   └── test_tier4_ir_benchmark.py  [KẾ THỪA] Sparse/Dense/RRF + end-to-end + đo độ trễ
│
├── tier5_generation_safety/        # Giai đoạn 3 — Tạo sinh
│   └── test_tier5_generation_safety.py [KẾ THỪA] RAG toàn trình, citation grounding, guardrail
│
└── rq_experiments_promptfoo/       # Giai đoạn 4 — Thực nghiệm 4 RQ (promptfoo, KHÔNG phải pytest)
    ├── providers/  provider.py | retrieval_provider.py | generation_provider.py   [KẾ THỪA, đã port path]
    ├── configs/    promptfooconfig.{baselines,retrieval,generation}.yaml          [KẾ THỪA, đã sửa path]
    ├── datasets/   demo.json (10 ca) | vietlawbench_promptfoo_tests.json (1000 ca) | jsonl_to_json.py  [KẾ THỪA]
    └── README.md   cách chạy từng RQ
```

Chú thích: **[KẾ THỪA]** = bê từ `promptfoo_demo` sang (đã sửa lại đường dẫn cho chạy được trong repo);
**[DỜI]** = di chuyển test cũ từ `tests/unit` & `tests/integration` vào đúng tầng; **[MỚI]** = viết mới để lấp
khoảng trống đặc tả.

## Ánh xạ file ↔ hạng mục đặc tả

| Tầng | File | Hạng mục đặc tả | RQ |
| --- | --- | --- | --- |
| 1 | test_tier1_data_curation, test_ast_parser, test_legal_chunker, test_text_cleaner | AST Parser, chunk theo đơn vị pháp lý | — |
| 1 | test_quality_text_gate, test_crawl_audit | Bộ lọc chất lượng (nguyên âm ≥0.05) | — |
| 2 | test_tier2_graph_multistore, test_hin_22_relations | Đồ thị HIN Neo4j 22 quan hệ, Qdrant | — |
| 2 | test_metadata_injection | Metadata Text Injection 11 trường | — |
| 2 | test_ingestion_stabilization, test_merge_crawl_artifacts | Pipeline nạp & đa kho | — |
| 3 | test_tier3_representation_mrl, test_mrl_loss, test_hierarchy_loss | Hierarchy-Aware Matryoshka InfoNCE | RQ2 |
| 3 | test_hin_triplet_miner | HIN-Guided Hard Negatives (metapath/RWR) | RQ1 |
| 4 | test_tier4_ir_benchmark | Truy xuất lai Dense/Sparse/RRF/Graph | RQ3 |
| 5 | test_tier5_generation_safety | Tạo sinh có căn cứ + guardrail | RQ4 |
| 4* | rq_experiments_promptfoo/* | Thực nghiệm số liệu 4 RQ (1000 câu) | RQ1–RQ4 |

## Cách chạy

**Chuẩn bị (một lần):** đặt biến môi trường nếu dữ liệu/checkpoint không nằm mặc định trong repo.
```bash
export VIETLAWBERT_ROOT=D:/UIT-HK7/VietLawBERT/vietlawbert   # mặc định: tự dò
export VIETLAWBERT_DATA=/duong/dan/toi/du_lieu               # mặc định: <repo>/data
export VIETLAWBERT_CKPT=/duong/dan/toi/checkpoints           # mặc định: <repo>/checkpoints
```

**pytest — kiểm định toàn vẹn 5 tầng (máy trạm CPU là đủ):**
```bash
cd <repo>
python -m pytest tests -v                 # toàn bộ
python -m pytest tests/tier1_data_curation # một tầng
python -m pytest tests -m "not requires_neo4j and not requires_qdrant and not requires_es"  # chỉ phần không cần service
python tests/run_audit.py                  # chạy tuần tự + báo cáo tổng
```
Các test cần Neo4j/Qdrant/Elasticsearch sẽ **tự SKIP** (không fail) nếu service chưa bật.
Stack đã có sẵn đầy đủ trong `docker-compose.yml` ở gốc repo (MongoDB, Qdrant, Elasticsearch, Neo4j+APOC, Redis). Bật phần cần cho test:
```bash
docker compose up -d qdrant elasticsearch neo4j
```
`conftest.py` tự nạp `.env` ở gốc repo nên khớp sẵn creds (Neo4j `neo4j/vietlawbert2026`, Qdrant 6333, ES 9200). Test nào thiếu service tương ứng sẽ tự **SKIP**.

**promptfoo — thực nghiệm RQ:** xem `rq_experiments_promptfoo/README.md`.

## Về lựa chọn framework
- **pytest**: đúng cho tier1–3 (logic thuần + invariant), chạy được trên CPU, tất định → nền CI.
- **promptfoo**: đúng cho RQ (so nhiều provider/model, LLM-as-a-Judge, bảng số liệu) nhưng end-to-end trên CPU rất chậm → nên chạy trên Cloud GPU.
- **Đề xuất tối ưu**:
  - Độ trễ P95/P99 (RQ3): dùng `pytest-benchmark` hoặc tự log percentile trong `benchmark/evaluate_rqs.py` thay vì chỉ ngưỡng SLA thô.
  - Faithfulness (RQ4): dùng thư viện `ragas` gốc để có điểm định lượng 0–1, thay vì chỉ `llm-rubric` pass/fail.
  - Thống kê (RQ1): giữ `scipy.stats` paired t-test (đã có trong benchmark/evaluate_rqs.py) nhưng hợp nhất 1 harness để tránh mâu thuẫn 2 bảng baseline.
