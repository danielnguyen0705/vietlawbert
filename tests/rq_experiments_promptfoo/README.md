# Thực nghiệm 4 RQ bằng promptfoo

> Đã port từ `promptfoo_demo`: bỏ hard-code `/home/dell/...`; đường dẫn `file://` trỏ sang
> `../providers` và `../datasets`. Checkpoint dùng đường dẫn tương đối `checkpoints/<backbone>`
> (provider tự neo về gốc repo hoặc đặt `VIETLAWBERT_CKPT`/`VIETLAWBERT_ROOT`).

## Chuẩn bị
- Bật service: Qdrant (6333), Elasticsearch (9200), Neo4j (7687) — `docker compose up -d`.
- Đặt khóa LLM judge/generator trong `.env` (xem phần Rào cản #2 trong báo cáo tiến độ: endpoint primary đang 403).
- `npm i -g promptfoo` (hoặc `npx promptfoo`).

## Lệnh chạy
```bash
cd tests/rq_experiments_promptfoo

# RQ2 — so sánh baseline đa backbone (zero-shot vs Ours-MRL@256), 1000 câu
promptfoo eval -c configs/promptfooconfig.baselines.yaml

# RQ1 & RQ3 — ablation truy xuất Dense/Sparse/RRF/Full
promptfoo eval -c configs/promptfooconfig.retrieval.yaml --table

# RQ4 — tạo sinh & độ trung thực (Faithfulness/Citation/Guardrail)
promptfoo eval -c configs/promptfooconfig.generation.yaml --table

promptfoo view   # xem kết quả
```

## Lưu ý vận hành (từ log 15/10)
- Chạy CPU rất chậm (timeout 300s/worker, cold-start cache đồ thị ~27s). Chạy thật nên để trên Cloud GPU; đã đặt `maxConcurrency: 1`, `timeoutMs: 600000`.
- Để chạy nhanh 10 câu thử: trỏ `tests:` sang `datasets/demo.json` (mặc định). Chạy đủ 1000 câu: đổi sang `datasets/vietlawbench_promptfoo_tests.json`.
- `provider.py` cache embedding tại `providers/.cache_promptfoo_embeddings/`.
