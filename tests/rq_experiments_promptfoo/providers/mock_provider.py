# -*- coding: utf-8 -*-
"""
mock_provider.py — Provider GIẢ cho promptfoo, để DEMO harness khi chưa bật backend thật.

Trả JSON đúng schema mà assertion kỳ vọng:
  {retrieved_chunks, retrieved_docs, latency_ms, answer}
Dựa trên ground-truth trong vars để ~70% câu "trúng" và ~30% "trượt" (có chủ đích),
cộng latency ngẫu nhiên khác nhau theo mode -> đủ để thấy metric & latency report hoạt động.

KHÔNG dùng cho số liệu báo cáo. Khi có provider thật, đổi config sang file provider thật.
"""
import json, hashlib, random

def _h(s):
    return int(hashlib.md5(str(s).encode("utf-8")).hexdigest(), 16)

def call_api(prompt, options, context):
    vars = (context or {}).get("vars", {}) or {}
    cfg = (options or {}).get("config", {}) or {}
    mode = cfg.get("mode", "dense")

    gt_chunk = vars.get("ground_truth_chunk", "")
    gt_doc   = vars.get("ground_truth_doc_number", "")
    art      = vars.get("article_id") or ""
    cl       = vars.get("clause_id") or ""

    # Tỉ lệ trúng theo mode (full > hybrid > dense > sparse) để ablation có hình dạng
    base = {"sparse": 4, "dense": 6, "hybrid": 7, "full": 8}.get(mode, 6)
    bid  = vars.get("query", "")
    hit  = (_h(bid + mode) % 10) < base

    rnd = random.Random(_h(bid + mode))
    if hit:
        retrieved_chunks = [gt_chunk, f"{gt_chunk.split('_')[0]}_art_99"]
        retrieved_docs   = [gt_doc, "99/2020/NĐ-CP"]
        answer = f"Theo {gt_doc}, {vars.get('ground_truth_article','')}: (nội dung trích dẫn)."
    else:
        retrieved_chunks = ["other_art_1_cl_1", "other_art_2"]
        retrieved_docs   = ["88/2019/NĐ-CP", "12/2018/TT-BTC"]
        answer = "Cơ sở dữ liệu không đủ căn cứ để trả lời."

    latency = {"sparse": rnd.uniform(80, 300),
               "dense":  rnd.uniform(120, 350),
               "hybrid": rnd.uniform(250, 1200),
               "full":   rnd.uniform(300, 27000)}.get(mode, 200)  # full có cold-start

    out = {
        "retrieved_chunks": retrieved_chunks,
        "retrieved_docs": retrieved_docs,
        "latency_ms": round(latency, 2),
        "answer": answer,
        "top_contexts": [answer],
    }
    return {"output": json.dumps(out, ensure_ascii=False)}
