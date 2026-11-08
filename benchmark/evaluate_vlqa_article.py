"""
benchmark/evaluate_vlqa_article.py — Đánh giá BM25 cấp điều luật trên tập dev VLQA (500 câu).

Chế độ:
- bm25_content: truy vấn chuẩn match content trên Elasticsearch index riêng vlqa_sparse_idx.
- system_sparse: truy vấn LegalElasticsearchRetriever.search_sparse (đa trường doc_number, hierarchy_path, content).
- bm25_plain: match content_plain (giữ dấu, không tách từ) trên vlqa_sparse_seg_idx.
- bm25_seg: tách từ câu hỏi bằng segmenter, sau đó match content_seg trên vlqa_sparse_seg_idx.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set

import numpy as np
from scipy import stats

from configs.config import config
from configs.paths import BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger
from benchmark.metrics_article import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    ap_at_k,
    f2_at_k,
    aggregate,
)
from preprocess.vi_segment import get_segmenter, segment_text

logger = get_subsystem_logger("benchmark", "evaluate_vlqa_article")


def check_reachability(es_client, index: str, questions: List[Dict[str, Any]]) -> Dict[str, int]:
    all_golds: Set[str] = set()
    for q in questions:
        all_golds.update(q.get("relevant_chunk_ids", []))

    if not all_golds:
        return {"missing_gold_ids": 0, "questions_partial_unreachable": 0, "questions_all_unreachable": 0}

    gold_list = list(all_golds)
    existing_ids = set()
    batch_size = 1000
    for i in range(0, len(gold_list), batch_size):
        chunk = gold_list[i : i + batch_size]
        res = es_client.mget(index=index, ids=chunk, _source=False)
        for doc in res.get("docs", []):
            if doc.get("found"):
                existing_ids.add(doc["_id"])

    missing_golds = all_golds - existing_ids
    q_partial = 0
    q_all = 0
    for q in questions:
        golds = set(q.get("relevant_chunk_ids", []))
        if not golds:
            continue
        missing_in_q = golds - existing_ids
        if len(missing_in_q) > 0:
            q_partial += 1
        if len(missing_in_q) == len(golds):
            q_all += 1

    stats_dict = {
        "total_gold_ids": len(all_golds),
        "existing_gold_ids": len(existing_ids),
        "missing_gold_ids": len(missing_golds),
        "questions_partial_unreachable": q_partial,
        "questions_all_unreachable": q_all,
    }
    return stats_dict


def deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    res = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            res.append(x)
    return res


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Đánh giá BM25 cấp điều luật trên tập dev VLQA")
    ap.add_argument("--dev", default=str(BENCHMARK_DIR / "vlqa_dev500.jsonl"))
    ap.add_argument("--index", default="vlqa_sparse_idx")
    ap.add_argument("--seg-index", default="vlqa_sparse_seg_idx")
    ap.add_argument("--segmenter", default="pyvi")
    ap.add_argument("--modes", default="bm25_content,system_sparse,bm25_plain,bm25_seg")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--ks", default="1,2,3,5,10")
    ap.add_argument("--out-dir", default=str(BENCHMARK_DIR / "results"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    main_es_idx = getattr(config, "ES_INDEX_NAME", "vietlaw_sparse_idx")
    if args.index == main_es_idx or args.seg_index == main_es_idx:
        logger.error("CẤM GHI ĐÈ/ĐỌC TRẦM: index trùng với config.ES_INDEX_NAME.")
        sys.exit(2)

    dev_path = Path(args.dev)
    if not dev_path.exists():
        logger.error("Không tìm thấy file dev: %s", dev_path)
        sys.exit(1)

    questions = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if args.limit > 0:
        questions = questions[: args.limit]
    logger.info("Nạp %d câu hỏi từ %s", len(questions), dev_path)

    from rag.es_retriever import LegalElasticsearchRetriever

    retriever = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=args.index)
    es_client = retriever.client

    reach_stats = check_reachability(es_client, args.index, questions)
    logger.info("=== KIỂM TRA ĐỘ BAO PHỦ GOLD TRONG CHỈ MỤC [%s] ===", args.index)
    logger.info("  Tổng số gold ids phân biệt        : %d", reach_stats.get("total_gold_ids", 0))
    logger.info("  Gold ids tồn tại trong index      : %d", reach_stats.get("existing_gold_ids", 0))
    logger.info("  Gold ids thiếu (không có trong ES): %d", reach_stats.get("missing_gold_ids", 0))
    logger.info("  Số câu có ≥ 1 gold thiếu          : %d", reach_stats.get("questions_partial_unreachable", 0))
    logger.info("  Số câu có TOÀN BỘ gold thiếu      : %d", reach_stats.get("questions_all_unreachable", 0))

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # Khởi tạo segmenter nếu có chế độ bm25_seg
    seg_fn = None
    if "bm25_seg" in modes:
        seg_fn = get_segmenter(args.segmenter)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    perquery_file = out_dir / "vlqa_dev500_perquery.jsonl"
    csv_file = out_dir / "vlqa_dev500_article_metrics.csv"
    sig_file = out_dir / "vlqa_dev500_significance.csv"

    if perquery_file.exists():
        perquery_file.unlink()

    all_mode_rows = []
    # Lưu per-query metrics để làm paired t-test: mode -> {qid: {"ndcg@10": float, "f2@2": float}}
    mode_query_metrics: Dict[str, Dict[Any, Dict[str, float]]] = {}

    for mode in modes:
        logger.info("--- Bắt đầu đánh giá chế độ: %s (depth=%d) ---", mode, args.depth)
        per_query_metrics = []
        latencies = []
        mode_query_metrics[mode] = {}

        with open(perquery_file, "a", encoding="utf-8") as pq_f:
            for idx, q in enumerate(questions, start=1):
                qid = q.get("qid")
                query_text = q.get("query", "")
                gold_ids = set(q.get("relevant_chunk_ids", []))

                t0 = time.perf_counter()
                if mode == "bm25_content":
                    body = {
                        "query": {"match": {"content": query_text}},
                        "size": args.depth,
                        "_source": ["chunk_id"],
                    }
                    resp = es_client.search(index=args.index, **body)
                    raw_ids = [hit["_id"] for hit in resp["hits"]["hits"]]
                elif mode == "system_sparse":
                    hits = retriever.search_sparse(query=query_text, top_k=args.depth, must_be_effective=True)
                    raw_ids = [h.get("chunk_id") or h.get("id") for h in hits]
                elif mode == "bm25_plain":
                    body = {
                        "query": {"match": {"content_plain": query_text}},
                        "size": args.depth,
                        "_source": ["chunk_id"],
                    }
                    resp = es_client.search(index=args.seg_index, **body)
                    raw_ids = [hit["_id"] for hit in resp["hits"]["hits"]]
                elif mode == "bm25_seg":
                    seg_query = segment_text(query_text, seg_fn)
                    body = {
                        "query": {"match": {"content_seg": seg_query}},
                        "size": args.depth,
                        "_source": ["chunk_id"],
                    }
                    resp = es_client.search(index=args.seg_index, **body)
                    raw_ids = [hit["_id"] for hit in resp["hits"]["hits"]]
                else:
                    logger.warning("Bỏ qua chế độ không xác định: %s", mode)
                    break
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                latencies.append(latency_ms)

                ranked = deduplicate_keep_order(raw_ids)

                q_metrics: Dict[str, float] = {}
                f2_dict: Dict[int, float] = {}
                for k in ks:
                    q_metrics[f"hit@{k}"] = hit_at_k(ranked, gold_ids, k)
                    q_metrics[f"recall@{k}"] = recall_at_k(ranked, gold_ids, k)
                    q_metrics[f"precision@{k}"] = precision_at_k(ranked, gold_ids, k)
                    f2_val = f2_at_k(ranked, gold_ids, k)
                    q_metrics[f"f2@{k}"] = f2_val
                    f2_dict[k] = round(f2_val, 4)

                q_metrics[f"recall@{args.depth}"] = recall_at_k(ranked, gold_ids, args.depth)

                ndcg10 = ndcg_at_k(ranked, gold_ids, 10)
                q_metrics["mrr@10"] = mrr_at_k(ranked, gold_ids, 10)
                q_metrics["ndcg@10"] = ndcg10
                q_metrics["ap@10"] = ap_at_k(ranked, gold_ids, 10)

                per_query_metrics.append(q_metrics)
                mode_query_metrics[mode][qid] = {
                    "ndcg@10": ndcg10,
                    "f2@2": q_metrics.get("f2@2", 0.0),
                }

                pq_record = {
                    "qid": qid,
                    "mode": mode,
                    "gold_ids": sorted(list(gold_ids)),
                    "top_10_ids": ranked[:10],
                    "ndcg@10": round(ndcg10, 4),
                    "f2": f2_dict,
                    "latency_ms": round(latency_ms, 2),
                }
                pq_f.write(json.dumps(pq_record, ensure_ascii=False) + chr(10))

                if idx % 100 == 0 or idx == len(questions):
                    logger.info("  [%s] Đã xử lý %d/%d câu (latency avg=%.1fms)", mode, idx, len(questions), sum(latencies)/len(latencies))

        agg = aggregate(per_query_metrics)
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)

        best_k = 1
        best_f2 = -1.0
        for k in ks:
            f2_k = agg.get(f"f2@{k}", 0.0)
            if f2_k > best_f2:
                best_f2 = f2_k
                best_k = k

        row = {
            "mode": mode,
            "n_queries": len(questions),
            "Hit@1": f"{agg.get('hit@1', 0.0):.4f}",
            "Hit@5": f"{agg.get('hit@5', 0.0):.4f}",
            "Hit@10": f"{agg.get('hit@10', 0.0):.4f}",
            "Recall@1": f"{agg.get('recall@1', 0.0):.4f}",
            "Recall@5": f"{agg.get('recall@5', 0.0):.4f}",
            "Recall@10": f"{agg.get('recall@10', 0.0):.4f}",
            f"Recall@{args.depth}": f"{agg.get(f'recall@{args.depth}', 0.0):.4f}",
            "MRR@10": f"{agg.get('mrr@10', 0.0):.4f}",
            "NDCG@10": f"{agg.get('ndcg@10', 0.0):.4f}",
            "MAP@10": f"{agg.get('ap@10', 0.0):.4f}",
            "Macro-F2@1": f"{agg.get('f2@1', 0.0):.4f}",
            "Macro-F2@2": f"{agg.get('f2@2', 0.0):.4f}",
            "Macro-F2@3": f"{agg.get('f2@3', 0.0):.4f}",
            "Macro-F2@5": f"{agg.get('f2@5', 0.0):.4f}",
            "Macro-F2@10": f"{agg.get('f2@10', 0.0):.4f}",
            "best_F2_k": best_k,
            "Latency_P50_ms": f"{p50:.2f}",
            "Latency_P95_ms": f"{p95:.2f}",
        }
        all_mode_rows.append(row)

    headers = [
        "mode",
        "n_queries",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        f"Recall@{args.depth}",
        "MRR@10",
        "NDCG@10",
        "MAP@10",
        "Macro-F2@1",
        "Macro-F2@2",
        "Macro-F2@3",
        "Macro-F2@5",
        "Macro-F2@10",
        "best_F2_k",
        "Latency_P50_ms",
        "Latency_P95_ms",
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=headers, lineterminator=chr(10))
        writer.writeheader()
        for r in all_mode_rows:
            writer.writerow(r)

    # 3. Tính Significance Test (Paired t-test vs bm25_content)
    baseline_mode = "bm25_content"
    sig_rows = []
    sig_headers = ["mode", "baseline", "metric", "mean_mode", "mean_baseline", "mean_diff", "t_stat", "p_value", "n"]

    if baseline_mode in mode_query_metrics:
        base_map = mode_query_metrics[baseline_mode]
        common_qids = [q.get("qid") for q in questions if q.get("qid") in base_map]

        for mode in modes:
            if mode == baseline_mode:
                continue
            cur_map = mode_query_metrics.get(mode, {})
            aligned_qids = [qid for qid in common_qids if qid in cur_map]
            n = len(aligned_qids)
            if n == 0:
                continue

            for metric in ["NDCG@10", "F2@2"]:
                key = "ndcg@10" if metric == "NDCG@10" else "f2@2"
                base_vals = [base_map[qid][key] for qid in aligned_qids]
                mode_vals = [cur_map[qid][key] for qid in aligned_qids]

                mean_mode = float(np.mean(mode_vals))
                mean_base = float(np.mean(base_vals))
                mean_diff = mean_mode - mean_base

                # Paired t-test
                ttest_res = stats.ttest_rel(mode_vals, base_vals)
                t_stat = float(ttest_res.statistic) if not np.isnan(ttest_res.statistic) else 0.0
                p_val = float(ttest_res.pvalue) if not np.isnan(ttest_res.pvalue) else 1.0

                sig_rows.append({
                    "mode": mode,
                    "baseline": baseline_mode,
                    "metric": metric,
                    "mean_mode": f"{mean_mode:.4f}",
                    "mean_baseline": f"{mean_base:.4f}",
                    "mean_diff": f"{mean_diff:.4f}",
                    "t_stat": f"{t_stat:.4f}",
                    "p_value": f"{p_val:.4f}",
                    "n": n,
                })

        with open(sig_file, "w", encoding="utf-8", newline="") as sf:
            swriter = csv.DictWriter(sf, fieldnames=sig_headers, lineterminator=chr(10))
            swriter.writeheader()
            for sr in sig_rows:
                swriter.writerow(sr)

    logger.info("✓ Đã lưu bảng kết quả metrics vào %s", csv_file)
    logger.info("✓ Đã lưu bảng kiểm định ý nghĩa thống kê vào %s", sig_file)
    logger.info("✓ Đã lưu nhật ký per-query vào %s", perquery_file)

    print(chr(10) + "=" * 90)
    print("KẾT QUẢ ĐÁNH GIÁ BASELINE BM25 CẤP ĐIỀU TRÊN TẬP DEV VLQA (500 CÂU):")
    print("=" * 90)
    with open(csv_file, "r", encoding="utf-8") as cf:
        print(cf.read().strip())
    print("=" * 90 + chr(10))

    if sig_file.exists():
        print("=" * 90)
        print("KẾT QUẢ KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (PAIRED T-TEST VS BM25_CONTENT):")
        print("=" * 90)
        with open(sig_file, "r", encoding="utf-8") as sf:
            print(sf.read().strip())
        print("=" * 90 + chr(10))


if __name__ == "__main__":
    main()
