"""
benchmark/evaluate_team_corpus.py — Đánh giá BM25 (giữ dấu và tách từ pyvi) trên kho văn bản nhóm
với bộ test VLQA (2.190 câu) và VietLawBench (710 câu mapped), tính điểm theo nhóm bản sao (group-aware).
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import numpy as np
from scipy import stats
from elasticsearch import Elasticsearch

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger
from preprocess.vi_segment import get_segmenter, segment_text
from benchmark.metrics_article import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    ap_at_k,
    f2_at_k,
)

logger = get_subsystem_logger("benchmark", "evaluate_team_corpus")

def to_group_ranking(ranked_chunk_ids: List[str], chunk_to_group: Dict[str, str]) -> List[str]:
    """
    Ánh xạ danh sách chunk_id sang group_id tương ứng.
    Chunk không có trong mapping giữ nguyên id.
    Chỉ giữ lần xuất hiện đầu tiên của mỗi group_id/chunk_id.
    """
    seen = set()
    res = []
    for cid in ranked_chunk_ids:
        gid = chunk_to_group.get(cid, cid)
        if gid not in seen:
            seen.add(gid)
            res.append(gid)
    return res

def evaluate_single_query(
    ranked_groups: List[str],
    gold_groups: Set[str],
    latency_ms: float,
) -> Dict[str, float]:
    """Tính các chỉ số đánh giá cho 1 câu hỏi."""
    return {
        "Hit@1": hit_at_k(ranked_groups, gold_groups, 1),
        "Hit@10": hit_at_k(ranked_groups, gold_groups, 10),
        "Recall@1": recall_at_k(ranked_groups, gold_groups, 1),
        "Recall@10": recall_at_k(ranked_groups, gold_groups, 10),
        "Recall@100": recall_at_k(ranked_groups, gold_groups, 100),
        "MRR@10": mrr_at_k(ranked_groups, gold_groups, 10),
        "NDCG@10": ndcg_at_k(ranked_groups, gold_groups, 10),
        "MAP@10": ap_at_k(ranked_groups, gold_groups, 10),
        "Macro-F2@1": f2_at_k(ranked_groups, gold_groups, 1),
        "Macro-F2@2": f2_at_k(ranked_groups, gold_groups, 2),
        "Macro-F2@3": f2_at_k(ranked_groups, gold_groups, 3),
        "Macro-F2@5": f2_at_k(ranked_groups, gold_groups, 5),
        "Macro-F2@10": f2_at_k(ranked_groups, gold_groups, 10),
        "latency_ms": latency_ms,
    }

def aggregate_subset_metrics(per_query_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """Tính trung bình các chỉ số và độ trễ P50, P95 cho 1 tập câu hỏi."""
    n = len(per_query_list)
    if n == 0:
        return {}

    metric_keys = [
        "Hit@1", "Hit@10", "Recall@1", "Recall@10", "Recall@100",
        "MRR@10", "NDCG@10", "MAP@10",
        "Macro-F2@1", "Macro-F2@2", "Macro-F2@3", "Macro-F2@5", "Macro-F2@10"
    ]

    res = {"n_queries": n}
    for k in metric_keys:
        vals = [q[k] for q in per_query_list]
        res[k] = round(float(np.mean(vals)), 4)

    lats = [q["latency_ms"] for q in per_query_list]
    res["Latency_P50_ms"] = round(float(np.percentile(lats, 50)), 2)
    res["Latency_P95_ms"] = round(float(np.percentile(lats, 95)), 2)
    return res

def run_paired_ttest(
    mode_vals: List[float],
    base_vals: List[float],
    dataset_name: str,
    mode_name: str,
    base_name: str,
    metric_name: str,
) -> Dict[str, Any]:
    """Thực hiện kiểm định t ghép cặp (paired t-test)."""
    n = len(mode_vals)
    arr_mode = np.array(mode_vals, dtype=np.float64)
    arr_base = np.array(base_vals, dtype=np.float64)
    diff = arr_mode - arr_base
    mean_diff = float(np.mean(diff))

    if np.allclose(arr_mode, arr_base):
        t_stat, p_val = 0.0, 1.0
    else:
        res = stats.ttest_rel(arr_mode, arr_base)
        t_stat, p_val = float(res.statistic), float(res.pvalue)

    return {
        "dataset": dataset_name,
        "mode": mode_name,
        "baseline": base_name,
        "metric": metric_name,
        "mean_mode": f"{float(np.mean(arr_mode)):.4f}",
        "mean_baseline": f"{float(np.mean(arr_base)):.4f}",
        "mean_diff": f"{mean_diff:+.4f}",
        "t_stat": f"{t_stat:.4f}",
        "p_value": f"{p_val:.4e}",
        "n": n,
    }

def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Đánh giá BM25 trên kho văn bản nhóm (VLQA và VietLawBench)")
    parser.add_argument("--index", default="team_sparse_seg_idx", help="Chỉ mục Elasticsearch")
    parser.add_argument("--vlqa-file", default=str(BENCHMARK_DIR / "vlqa_test2190_team.jsonl"))
    parser.add_argument("--vlb-file", default=str(BENCHMARK_DIR / "vietlawbench_tune_team.jsonl"))
    parser.add_argument("--out-dir", default=str(BENCHMARK_DIR / "results"))
    parser.add_argument("--perquery-out", default=r"D:\VLQA\task9\perquery_team_bm25.jsonl")
    parser.add_argument("--depth", type=int, default=100)
    args = parser.parse_args(argv)

    es = Elasticsearch([config.ES_HOST], timeout=60)
    seg_fn = get_segmenter("pyvi")

    perquery_path = Path(args.perquery_out)
    perquery_path.parent.mkdir(parents=True, exist_ok=True)
    perquery_file = open(perquery_path, "w", encoding="utf-8")

    # =========================================================================
    # PHẦN 1: ĐÁNH GIÁ VLQA (2.190 CÂU HỎI)
    # =========================================================================
    logger.info("=== ĐÁNH GIÁ VLQA TEST SET (2.190 CÂU HỎI) ===")
    vlqa_items = []
    with open(args.vlqa_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                vlqa_items.append(json.loads(line))
    logger.info("Đã nạp %d câu hỏi VLQA từ %s", len(vlqa_items), args.vlqa_file)

    vlqa_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "bm25_plain": {"all_2190": [], "reachable_full": [], "dev500_all": []},
        "bm25_seg": {"all_2190": [], "reachable_full": [], "dev500_all": []},
    }

    vlqa_qid_metrics: Dict[str, Dict[int, Dict[str, float]]] = {
        "bm25_plain": {},
        "bm25_seg": {},
    }

    vlqa_ceiling_count = 0

    for idx, item in enumerate(vlqa_items):
        qid = item["qid"]
        query = item["query"]
        orig_split = item.get("orig_split", "")
        status = item.get("status", "")
        per_aid = item.get("per_aid", {})

        chunk_to_group: Dict[str, str] = {}
        gold_groups: Set[str] = set()

        for aid_str, ainfo in per_aid.items():
            gid = f"aid:{aid_str}"
            gold_groups.add(gid)
            for cid in ainfo.get("gold_chunks", []):
                chunk_to_group[cid] = gid

        # Mode bm25_plain
        t0 = time.perf_counter()
        resp_plain = es.search(
            index=args.index,
            query={"match": {"content_plain": query}},
            size=args.depth,
            _source=["chunk_id"],
        )
        lat_plain = (time.perf_counter() - t0) * 1000.0
        cids_plain = [h["_source"]["chunk_id"] for h in resp_plain["hits"]["hits"]]
        ranked_plain = to_group_ranking(cids_plain, chunk_to_group)
        m_plain = evaluate_single_query(ranked_plain, gold_groups, lat_plain)

        vlqa_results["bm25_plain"]["all_2190"].append(m_plain)
        if status == "full":
            vlqa_results["bm25_plain"]["reachable_full"].append(m_plain)
        if orig_split == "dev":
            vlqa_results["bm25_plain"]["dev500_all"].append(m_plain)
        vlqa_qid_metrics["bm25_plain"][qid] = m_plain

        perquery_file.write(json.dumps({
            "dataset": "vlqa",
            "qid": qid,
            "mode": "bm25_plain",
            "gold_groups": sorted(list(gold_groups)),
            "retrieved_cids_top10": cids_plain[:10],
            "ranked_groups_top10": ranked_plain[:10],
            "ndcg@10": m_plain["NDCG@10"],
            "f2@2": m_plain["Macro-F2@2"],
            "latency_ms": round(lat_plain, 2),
        }, ensure_ascii=False) + "\n")

        # Mode bm25_seg
        t0_seg = time.perf_counter()
        seg_query = segment_text(query, seg_fn)
        resp_seg = es.search(
            index=args.index,
            query={"match": {"content_seg": seg_query}},
            size=args.depth,
            _source=["chunk_id"],
        )
        lat_seg = (time.perf_counter() - t0_seg) * 1000.0
        cids_seg = [h["_source"]["chunk_id"] for h in resp_seg["hits"]["hits"]]
        ranked_seg = to_group_ranking(cids_seg, chunk_to_group)
        m_seg = evaluate_single_query(ranked_seg, gold_groups, lat_seg)

        vlqa_results["bm25_seg"]["all_2190"].append(m_seg)
        if status == "full":
            vlqa_results["bm25_seg"]["reachable_full"].append(m_seg)
        if orig_split == "dev":
            vlqa_results["bm25_seg"]["dev500_all"].append(m_seg)
        vlqa_qid_metrics["bm25_seg"][qid] = m_seg

        if m_seg["Recall@100"] > 0:
            vlqa_ceiling_count += 1

        perquery_file.write(json.dumps({
            "dataset": "vlqa",
            "qid": qid,
            "mode": "bm25_seg",
            "gold_groups": sorted(list(gold_groups)),
            "retrieved_cids_top10": cids_seg[:10],
            "ranked_groups_top10": ranked_seg[:10],
            "ndcg@10": m_seg["NDCG@10"],
            "f2@2": m_seg["Macro-F2@2"],
            "latency_ms": round(lat_seg, 2),
        }, ensure_ascii=False) + "\n")

        if (idx + 1) % 500 == 0 or (idx + 1) == len(vlqa_items):
            logger.info("  VLQA: Đã đánh giá %d/%d câu hỏi...", idx + 1, len(vlqa_items))

    # =========================================================================
    # PHẦN 2: ĐÁNH GIÁ VIETLAWBENCH (710 CÂU HỎI MAPPED)
    # =========================================================================
    logger.info("=== ĐÁNH GIÁ VIETLAWBENCH (710 CÂU HỎI MAPPED) ===")
    vlb_items = []
    with open(args.vlb_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "mapped":
                    vlb_items.append(row)
    logger.info("Đã nạp %d câu hỏi VietLawBench mapped từ %s", len(vlb_items), args.vlb_file)

    vlb_results: Dict[str, List[Dict[str, float]]] = {
        "bm25_plain": [],
        "bm25_seg": [],
    }
    vlb_id_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "bm25_plain": {},
        "bm25_seg": {},
    }

    for idx, item in enumerate(vlb_items):
        bid = item["benchmark_id"]
        query = item["query"]
        relevant_cids = item.get("relevant_chunk_ids", [])

        gid = f"vlb:{bid}"
        chunk_to_group = {cid: gid for cid in relevant_cids}
        gold_groups = {gid}

        # Mode bm25_plain
        t0 = time.perf_counter()
        resp_plain = es.search(
            index=args.index,
            query={"match": {"content_plain": query}},
            size=args.depth,
            _source=["chunk_id"],
        )
        lat_plain = (time.perf_counter() - t0) * 1000.0
        cids_plain = [h["_source"]["chunk_id"] for h in resp_plain["hits"]["hits"]]
        ranked_plain = to_group_ranking(cids_plain, chunk_to_group)
        m_plain = evaluate_single_query(ranked_plain, gold_groups, lat_plain)

        vlb_results["bm25_plain"].append(m_plain)
        vlb_id_metrics["bm25_plain"][bid] = m_plain

        perquery_file.write(json.dumps({
            "dataset": "vietlawbench",
            "benchmark_id": bid,
            "mode": "bm25_plain",
            "gold_groups": [gid],
            "retrieved_cids_top10": cids_plain[:10],
            "ranked_groups_top10": ranked_plain[:10],
            "ndcg@10": m_plain["NDCG@10"],
            "f2@2": m_plain["Macro-F2@2"],
            "latency_ms": round(lat_plain, 2),
        }, ensure_ascii=False) + "\n")

        # Mode bm25_seg
        t0_seg = time.perf_counter()
        seg_query = segment_text(query, seg_fn)
        resp_seg = es.search(
            index=args.index,
            query={"match": {"content_seg": seg_query}},
            size=args.depth,
            _source=["chunk_id"],
        )
        lat_seg = (time.perf_counter() - t0_seg) * 1000.0
        cids_seg = [h["_source"]["chunk_id"] for h in resp_seg["hits"]["hits"]]
        ranked_seg = to_group_ranking(cids_seg, chunk_to_group)
        m_seg = evaluate_single_query(ranked_seg, gold_groups, lat_seg)

        vlb_results["bm25_seg"].append(m_seg)
        vlb_id_metrics["bm25_seg"][bid] = m_seg

        perquery_file.write(json.dumps({
            "dataset": "vietlawbench",
            "benchmark_id": bid,
            "mode": "bm25_seg",
            "gold_groups": [gid],
            "retrieved_cids_top10": cids_seg[:10],
            "ranked_groups_top10": ranked_seg[:10],
            "ndcg@10": m_seg["NDCG@10"],
            "f2@2": m_seg["Macro-F2@2"],
            "latency_ms": round(lat_seg, 2),
        }, ensure_ascii=False) + "\n")

        if (idx + 1) % 200 == 0 or (idx + 1) == len(vlb_items):
            logger.info("  VLB: Đã đánh giá %d/%d câu hỏi...", idx + 1, len(vlb_items))

    perquery_file.close()
    logger.info("✓ Đã lưu chi tiết từng câu hỏi vào: %s", perquery_path)

    # =========================================================================
    # PHẦN 3: XUẤT CSV KẾT QUẢ VÀ TỔNG KẾT
    # =========================================================================
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_headers = [
        "mode", "subset", "n_queries",
        "Hit@1", "Hit@10", "Recall@1", "Recall@10", "Recall@100",
        "MRR@10", "NDCG@10", "MAP@10",
        "Macro-F2@1", "Macro-F2@2", "Macro-F2@3", "Macro-F2@5", "Macro-F2@10",
        "Latency_P50_ms", "Latency_P95_ms"
    ]

    # 1. VLQA CSV
    vlqa_csv_path = out_dir / "team_vlqa2190_metrics.csv"
    vlqa_rows = []
    for mode in ["bm25_plain", "bm25_seg"]:
        for subset in ["all_2190", "reachable_full", "dev500_all"]:
            q_list = vlqa_results[mode][subset]
            agg = aggregate_subset_metrics(q_list)
            row = {"mode": mode, "subset": subset}
            row.update(agg)
            vlqa_rows.append(row)

    with open(vlqa_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(vlqa_rows)
    logger.info("✓ Đã lưu bảng kết quả VLQA vào: %s", vlqa_csv_path)

    # 2. VietLawBench CSV
    vlb_csv_path = out_dir / "team_vlb710_metrics.csv"
    vlb_rows = []
    for mode in ["bm25_plain", "bm25_seg"]:
        q_list = vlb_results[mode]
        agg = aggregate_subset_metrics(q_list)
        row = {"mode": mode, "subset": "mapped_710"}
        row.update(agg)
        vlb_rows.append(row)

    with open(vlb_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(vlb_rows)
    logger.info("✓ Đã lưu bảng kết quả VietLawBench vào: %s", vlb_csv_path)

    # 3. Significance CSV
    sig_csv_path = out_dir / "team_bm25_significance.csv"
    sig_headers = ["dataset", "mode", "baseline", "metric", "mean_mode", "mean_baseline", "mean_diff", "t_stat", "p_value", "n"]
    sig_rows = []

    # VLQA all_2190
    vlqa_qids = sorted(list(vlqa_qid_metrics["bm25_seg"].keys()))
    for met in ["NDCG@10", "Macro-F2@2"]:
        seg_vals = [vlqa_qid_metrics["bm25_seg"][q][met] for q in vlqa_qids]
        base_vals = [vlqa_qid_metrics["bm25_plain"][q][met] for q in vlqa_qids]
        sig_rows.append(run_paired_ttest(
            seg_vals, base_vals,
            dataset_name="vlqa_all_2190",
            mode_name="bm25_seg",
            base_name="bm25_plain",
            metric_name="NDCG@10" if met == "NDCG@10" else "F2@2",
        ))

    # VietLawBench mapped_710
    vlb_bids = sorted(list(vlb_id_metrics["bm25_seg"].keys()))
    for met in ["NDCG@10", "Macro-F2@2"]:
        seg_vals = [vlb_id_metrics["bm25_seg"][b][met] for b in vlb_bids]
        base_vals = [vlb_id_metrics["bm25_plain"][b][met] for b in vlb_bids]
        sig_rows.append(run_paired_ttest(
            seg_vals, base_vals,
            dataset_name="vlb_mapped_710",
            mode_name="bm25_seg",
            base_name="bm25_plain",
            metric_name="NDCG@10" if met == "NDCG@10" else "F2@2",
        ))

    with open(sig_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sig_headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sig_rows)
    logger.info("✓ Đã lưu kiểm định ý nghĩa thống kê vào: %s", sig_csv_path)

    # In kết quả console
    print("\n" + "=" * 95)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ TRÊN KHO VĂN BẢN NHÓM - VLQA TEST SET (team_vlqa2190_metrics.csv):")
    print("=" * 95)
    with open(vlqa_csv_path, "r", encoding="utf-8") as f:
        print(f.read().strip())
    print("=" * 95)

    print("\n" + "=" * 95)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ TRÊN KHO VĂN BẢN NHÓM - VIETLAWBENCH (team_vlb710_metrics.csv):")
    print("=" * 95)
    with open(vlb_csv_path, "r", encoding="utf-8") as f:
        print(f.read().strip())
    print("=" * 95)

    print("\n" + "=" * 95)
    print("BẢNG KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (team_bm25_significance.csv):")
    print("=" * 95)
    with open(sig_csv_path, "r", encoding="utf-8") as f:
        print(f.read().strip())
    print("=" * 95)

    # Candidate ceiling
    total_vlqa = len(vlqa_items)
    print(f"\n[CANDIDATE CEILING] VLQA all_2190: {vlqa_ceiling_count}/{total_vlqa} câu hỏi ({vlqa_ceiling_count/total_vlqa*100:.2f}%) có ít nhất 1 gold group trong top-100 của bm25_seg.")
    print(f"Tổng thời gian chạy đánh giá: {time.perf_counter() - t_start:.2f} giây\n")

if __name__ == "__main__":
    main()