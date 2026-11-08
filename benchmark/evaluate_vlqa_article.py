"""
benchmark/evaluate_vlqa_article.py — Đánh giá BM25, Dense và Hybrid RRF cấp điều luật trên tập dev VLQA (500 câu).

Chế độ:
- bm25_content: truy vấn match content trên Elasticsearch index riêng vlqa_sparse_idx.
- system_sparse: truy vấn LegalElasticsearchRetriever.search_sparse.
- bm25_plain: match content_plain (giữ dấu) trên vlqa_sparse_seg_idx.
- bm25_seg: tách từ câu hỏi bằng pyvi, match content_seg trên vlqa_sparse_seg_idx.
- dense: tìm kiếm vector precomputed BAAI/bge-m3 trong Qdrant vlqa_dense_bge_m3.
- hybrid_rrf: kết hợp xếp hạng (RRF) giữa bm25_seg và dense.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import httpx
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


def aggregate_pieces(hits: List[Tuple[str, float]]) -> List[str]:
    """
    hits là danh sách (chunk_id, score) sắp xếp theo score giảm dần.
    Giữ lần xuất hiện ĐẦU TIÊN của mỗi chunk_id (= max score cho mỗi bài/điều).
    """
    seen = set()
    res = []
    for chunk_id, _score in hits:
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            res.append(chunk_id)
    return res


def rrf_fuse(rankings: List[List[str]], k: int = 60, depth: int = 100) -> List[str]:
    """
    RRF Score: score(c) = sum(1 / (k + rank_c)) với rank bắt đầu từ 1.
    Sắp xếp theo score giảm dần, hòa điểm (tie) xếp theo chunk_id tăng dần.
    Trả về top depth chunk_id.
    """
    scores: Dict[str, float] = {}
    for r_list in rankings:
        for rank, cid in enumerate(r_list, start=1):
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [cid for cid, _score in sorted_items[:depth]]


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

    raw_vlqa_dir = os.environ.get("VLQA_DIR") or getattr(config, "VLQA_DIR", "")
    vlqa_dir_default = raw_vlqa_dir if raw_vlqa_dir else "D:\\VLQA"
    default_vectors_dir = str(Path(vlqa_dir_default) / "vectors" / "bge-m3")

    ap = argparse.ArgumentParser(description="Đánh giá BM25, Dense và Hybrid RRF cấp điều luật trên tập dev VLQA")
    ap.add_argument("--dev", default=str(BENCHMARK_DIR / "vlqa_dev500.jsonl"))
    ap.add_argument("--index", default="vlqa_sparse_idx")
    ap.add_argument("--seg-index", default="vlqa_sparse_seg_idx")
    ap.add_argument("--segmenter", default="pyvi")
    ap.add_argument("--dense-collection", default="vlqa_dense_bge_m3")
    ap.add_argument("--vectors-dir", default=default_vectors_dir)
    ap.add_argument("--piece-depth", type=int, default=400)
    ap.add_argument("--exact", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rrf-k", type=int, default=getattr(config, "RRF_K", 60))
    ap.add_argument("--baselines", default="bm25_content,bm25_seg")
    ap.add_argument("--modes", default="bm25_content,system_sparse,bm25_plain,bm25_seg,dense,hybrid_rrf")
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

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    needs_es = any(m in modes for m in ["bm25_content", "system_sparse", "bm25_plain", "bm25_seg", "hybrid_rrf"])

    es_retriever = None
    es_client = None
    if needs_es:
        from rag.es_retriever import LegalElasticsearchRetriever
        es_retriever = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=args.index)
        es_client = es_retriever.client

        reach_stats = check_reachability(es_client, args.index, questions)
        logger.info("=== KIỂM TRA ĐỘ BAO PHỦ GOLD TRONG CHỈ MỤC [%s] ===", args.index)
        logger.info("  Tổng số gold ids phân biệt        : %d", reach_stats.get("total_gold_ids", 0))
        logger.info("  Gold ids tồn tại trong index      : %d", reach_stats.get("existing_gold_ids", 0))
        logger.info("  Gold ids thiếu (không có trong ES): %d", reach_stats.get("missing_gold_ids", 0))
        logger.info("  Số câu có ≥ 1 gold thiếu          : %d", reach_stats.get("questions_partial_unreachable", 0))
        logger.info("  Số câu có TOÀN BỘ gold thiếu      : %d", reach_stats.get("questions_all_unreachable", 0))

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    # Khởi tạo segmenter nếu có chế độ bm25_seg hoặc hybrid_rrf
    seg_fn = None
    if any(m in modes for m in ["bm25_seg", "hybrid_rrf"]):
        seg_fn = get_segmenter(args.segmenter)

    # Khởi tạo HTTP Client cho Qdrant
    qdrant_http = None
    query_vectors = None
    qid_to_vec_idx = {}
    needs_dense = any(m in modes for m in ["dense", "hybrid_rrf"])

    if needs_dense:
        qdrant_url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
        qdrant_http = httpx.Client(base_url=qdrant_url, timeout=120.0)

        vectors_dir = Path(args.vectors_dir)
        qvec_path = vectors_dir / "query_vectors.npy"
        qids_path = vectors_dir / "query_qids.json"
        if not qvec_path.exists() or not qids_path.exists():
            logger.error("Không tìm thấy file query vectors: %s hoặc %s", qvec_path, qids_path)
            sys.exit(1)

        query_vectors = np.load(qvec_path)
        with open(qids_path, "r", encoding="utf-8") as qf:
            qids_list = json.load(qf)
        qid_to_vec_idx = {qid: i for i, qid in enumerate(qids_list)}
        logger.info("Nạp %d query vectors từ %s", len(query_vectors), qvec_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    perquery_file = out_dir / "vlqa_dev500_perquery.jsonl"
    csv_file = out_dir / "vlqa_dev500_article_metrics.csv"
    sig_file = out_dir / "vlqa_dev500_significance.csv"

    if perquery_file.exists():
        perquery_file.unlink()

    all_mode_rows = []
    mode_query_metrics: Dict[str, Dict[Any, Dict[str, float]]] = {}
    mode_agg: Dict[str, Dict[str, float]] = {}

    bm25_seg_cache: Dict[Any, Tuple[List[str], float]] = {}
    dense_cache: Dict[Any, Tuple[List[str], float]] = {}

    latency_scopes = {
        "bm25_content": "search_only",
        "system_sparse": "search_only",
        "bm25_plain": "search_only",
        "dense": "search_only",
        "bm25_seg": "includes_query_segmentation",
        "hybrid_rrf": "bm25_seg+dense, query vector precomputed",
    }

    def execute_dense_search(qid: Any) -> Tuple[List[str], float]:
        if qid not in qid_to_vec_idx:
            logger.error("Không tìm thấy vector cho qid %s trong query_qids.json", qid)
            return [], 0.0
        vec_idx = qid_to_vec_idx[qid]
        qvec = query_vectors[vec_idx].astype(np.float32).tolist()

        t0 = time.perf_counter()
        body = {
            "vector": qvec,
            "limit": args.piece_depth,
            "with_payload": ["chunk_id"],
            "params": {"exact": args.exact},
        }
        resp = qdrant_http.post(f"/collections/{args.dense_collection}/points/search", json=body)
        resp.raise_for_status()
        raw_hits = resp.json().get("result", [])
        hits = [(h["payload"]["chunk_id"], h["score"]) for h in raw_hits if "payload" in h and "chunk_id" in h["payload"]]
        ranked_dense = aggregate_pieces(hits)[: args.depth]
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return ranked_dense, lat_ms

    def execute_bm25_seg_search(query_text: str) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        seg_query = segment_text(query_text, seg_fn)
        body = {
            "query": {"match": {"content_seg": seg_query}},
            "size": args.depth,
            "_source": ["chunk_id"],
        }
        resp = es_client.search(index=args.seg_index, **body)
        raw_ids = [hit["_id"] for hit in resp["hits"]["hits"]]
        ranked_seg = deduplicate_keep_order(raw_ids)[: args.depth]
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return ranked_seg, lat_ms

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

                if mode == "bm25_content":
                    t0 = time.perf_counter()
                    body = {
                        "query": {"match": {"content": query_text}},
                        "size": args.depth,
                        "_source": ["chunk_id"],
                    }
                    resp = es_client.search(index=args.index, **body)
                    raw_ids = [hit["_id"] for hit in resp["hits"]["hits"]]
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    ranked = deduplicate_keep_order(raw_ids)[: args.depth]

                elif mode == "system_sparse":
                    t0 = time.perf_counter()
                    hits = es_retriever.search_sparse(query=query_text, top_k=args.depth, must_be_effective=True)
                    raw_ids = [h.get("chunk_id") or h.get("id") for h in hits]
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    ranked = deduplicate_keep_order(raw_ids)[: args.depth]

                elif mode == "bm25_plain":
                    t0 = time.perf_counter()
                    body = {
                        "query": {"match": {"content_plain": query_text}},
                        "size": args.depth,
                        "_source": ["chunk_id"],
                    }
                    resp = es_client.search(index=args.seg_index, **body)
                    raw_ids = [hit["_id"] for hit in resp["hits"]["hits"]]
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    ranked = deduplicate_keep_order(raw_ids)[: args.depth]

                elif mode == "bm25_seg":
                    ranked, latency_ms = execute_bm25_seg_search(query_text)
                    bm25_seg_cache[qid] = (ranked, latency_ms)

                elif mode == "dense":
                    ranked, latency_ms = execute_dense_search(qid)
                    dense_cache[qid] = (ranked, latency_ms)

                elif mode == "hybrid_rrf":
                    if qid in bm25_seg_cache:
                        seg_ranked, seg_lat = bm25_seg_cache[qid]
                    else:
                        seg_ranked, seg_lat = execute_bm25_seg_search(query_text)
                        bm25_seg_cache[qid] = (seg_ranked, seg_lat)

                    if qid in dense_cache:
                        den_ranked, den_lat = dense_cache[qid]
                    else:
                        den_ranked, den_lat = execute_dense_search(qid)
                        dense_cache[qid] = (den_ranked, den_lat)

                    ranked = rrf_fuse([seg_ranked[:100], den_ranked[:100]], k=args.rrf_k, depth=args.depth)
                    latency_ms = seg_lat + den_lat

                else:
                    logger.warning("Bỏ qua chế độ không xác định: %s", mode)
                    break

                latencies.append(latency_ms)

                q_metrics = {}
                for k in ks:
                    q_metrics[f"hit@{k}"] = hit_at_k(ranked, gold_ids, k)
                    q_metrics[f"recall@{k}"] = recall_at_k(ranked, gold_ids, k)
                    q_metrics[f"precision@{k}"] = precision_at_k(ranked, gold_ids, k)
                    q_metrics[f"f2@{k}"] = f2_at_k(ranked, gold_ids, k)

                q_metrics[f"recall@{args.depth}"] = recall_at_k(ranked, gold_ids, args.depth)
                q_metrics["mrr@10"] = mrr_at_k(ranked, gold_ids, 10)
                q_metrics["ndcg@10"] = ndcg_at_k(ranked, gold_ids, 10)
                q_metrics["ap@10"] = ap_at_k(ranked, gold_ids, 10)

                per_query_metrics.append(q_metrics)

                ndcg10 = q_metrics["ndcg@10"]
                f2_2 = q_metrics.get("f2@2", 0.0)
                mode_query_metrics[mode][qid] = {
                    "ndcg@10": ndcg10,
                    "f2@2": f2_2,
                }

                f2_dict = {f"k={k}": round(q_metrics[f"f2@{k}"], 4) for k in ks}
                pq_record = {
                    "qid": qid,
                    "mode": mode,
                    "gold_ids": sorted(list(gold_ids)),
                    "top_10_ids": ranked[:10],
                    "ndcg@10": round(ndcg10, 4),
                    "f2": f2_dict,
                    "latency_ms": round(latency_ms, 2),
                }
                pq_f.write(json.dumps(pq_record, ensure_ascii=False) + "\n")

                if idx % 100 == 0 or idx == len(questions):
                    logger.info("  [%s] Đã xử lý %d/%d câu (latency avg=%.1fms)", mode, idx, len(questions), sum(latencies)/len(latencies))

        agg = aggregate(per_query_metrics)
        mode_agg[mode] = agg
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
            "Latency_scope": latency_scopes.get(mode, "search_only"),
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
        "Latency_scope",
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for r in all_mode_rows:
            writer.writerow(r)

    # 3. Tính Significance Test (Paired t-test vs từng baseline trong args.baselines)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    sig_rows = []
    sig_headers = ["mode", "baseline", "metric", "mean_mode", "mean_baseline", "mean_diff", "t_stat", "p_value", "n"]

    for baseline_mode in baselines:
        if baseline_mode not in mode_query_metrics:
            continue
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
        swriter = csv.DictWriter(sf, fieldnames=sig_headers, lineterminator="\n")
        swriter.writeheader()
        for sr in sig_rows:
            swriter.writerow(sr)

    # 4. Cross-check Dense vs sanity_metrics.json
    if "dense" in mode_agg:
        sanity_file = Path(args.vectors_dir) / "sanity_metrics.json"
        if sanity_file.exists():
            with open(sanity_file, "r", encoding="utf-8") as sf:
                sanity_data = json.load(sf)
            dense_agg = mode_agg["dense"]
            r10_act = dense_agg.get("recall@10", 0.0)
            ndcg10_act = dense_agg.get("ndcg@10", 0.0)
            f2_2_act = dense_agg.get("f2@2", 0.0)

            r10_exp = sanity_data.get("Recall@10", 0.0)
            ndcg10_exp = sanity_data.get("NDCG@10", 0.0)
            f2_2_exp = sanity_data.get("F2@2", 0.0)

            diff_r10 = abs(r10_act - r10_exp)
            diff_ndcg10 = abs(ndcg10_act - ndcg10_exp)
            diff_f2_2 = abs(f2_2_act - f2_2_exp)

            print("\n" + "=" * 90)
            print("KIỂM TRA CHÉO (CROSS-CHECK) DENSE VS SANITY_METRICS.JSON:")
            print(f"  Recall@10 : thực tế={r10_act:.4f}, sanity={r10_exp:.4f} (lệch={diff_r10:.4f})")
            print(f"  NDCG@10   : thực tế={ndcg10_act:.4f}, sanity={ndcg10_exp:.4f} (lệch={diff_ndcg10:.4f})")
            print(f"  F2@2      : thực tế={f2_2_act:.4f}, sanity={f2_2_exp:.4f} (lệch={diff_f2_2:.4f})")
            print("=" * 90 + "\n")

            if args.exact and len(questions) == sanity_data.get("n_queries", 500):
                if diff_r10 > 0.01 or diff_ndcg10 > 0.01 or diff_f2_2 > 0.01:
                    logger.error("LỆCH LỚN HƠN 0.01: Có thể có lỗi aggregation hoặc lệch qid!")
                    sys.exit(1)
                else:
                    logger.info("✓ Cross-check thành công! Độ lệch đều <= 0.01.")

    if qdrant_http is not None:
        qdrant_http.close()

    logger.info("✓ Đã lưu bảng kết quả metrics vào %s", csv_file)
    logger.info("✓ Đã lưu bảng kiểm định ý nghĩa thống kê vào %s", sig_file)
    logger.info("✓ Đã lưu nhật ký per-query vào %s", perquery_file)

    print("\n" + "=" * 90)
    print("KẾT QUẢ ĐÁNH GIÁ CẤP ĐIỀU TRÊN TẬP DEV VLQA (500 CÂU):")
    print("=" * 90)
    with open(csv_file, "r", encoding="utf-8") as cf:
        print(cf.read().strip())
    print("=" * 90 + "\n")

    if sig_file.exists():
        print("=" * 90)
        print("KẾT QUẢ KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (PAIRED T-TEST):")
        print("=" * 90)
        with open(sig_file, "r", encoding="utf-8") as sf:
            print(sf.read().strip())
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()