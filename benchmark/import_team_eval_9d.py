"""
benchmark/import_team_eval_9d.py — Nhập và tính lại kết quả Dense, Hybrid RRF k=60,
Rerank Top-50 trên kho văn bản nhóm (tập test VLQA 2.190 và VietLawBench 710),
đồng thời hỗ trợ đánh giá cắt bỏ với analyzer cũ (team_sparse_legacy_idx).
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

import numpy as np
from configs.config import config
from configs.paths import BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger
from benchmark.metrics_article import (
    hit_at_k,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    ap_at_k,
    f2_at_k,
    aggregate,
)
from benchmark.evaluate_team_corpus import to_group_ranking, run_paired_ttest
from benchmark.evaluate_vlqa_article import rrf_fuse
from benchmark.tune_vlqa_fusion import order_rerank_candidates

logger = get_subsystem_logger("benchmark", "import_team_eval_9d")

METRIC_KEYS = [
    "Hit@1", "Hit@10", "Recall@1", "Recall@10", "Recall@50", "Recall@100",
    "MRR@10", "NDCG@10", "MAP@10", "F2@2", "F2@3",
]


def deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    res = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            res.append(x)
    return res


def verify_rrf_candidates(
    bm25_map: Dict[Any, List[str]],
    dense_map: Dict[Any, List[str]],
    rerank_candidates_map: Dict[Any, List[str]],
    k: int = 60,
    depth: int = 100,
    top_n: int = 50,
) -> Tuple[Dict[Any, List[str]], List[Any]]:
    """
    Tính lại RRF k=60 cục bộ từ BM25 và Dense, đối chiếu top_n đầu tiên
    với danh sách candidates đã gửi cho reranker trên Kaggle.
    Trả về (hybrid_map, danh sách khóa không khớp).
    """
    hybrid_map: Dict[Any, List[str]] = {}
    mismatches: List[Any] = []
    for key, expected_cands in rerank_candidates_map.items():
        b_ranked = bm25_map[key]
        d_ranked = dense_map[key]
        fused = rrf_fuse([b_ranked, d_ranked], k=k, depth=depth)
        hybrid_map[key] = fused
        if fused[:top_n] != expected_cands:
            mismatches.append(key)
    return hybrid_map, mismatches


def evaluate_ranking(
    ranked_chunk_ids: List[str],
    chunk_to_group: Dict[str, str],
    gold_groups: Set[str],
) -> Dict[str, float]:
    """Đánh giá theo nhóm bản sao (group-aware) cho 1 truy vấn."""
    ranked_groups = to_group_ranking(ranked_chunk_ids, chunk_to_group)
    return {
        "Hit@1": hit_at_k(ranked_groups, gold_groups, 1),
        "Hit@10": hit_at_k(ranked_groups, gold_groups, 10),
        "Recall@1": recall_at_k(ranked_groups, gold_groups, 1),
        "Recall@10": recall_at_k(ranked_groups, gold_groups, 10),
        "Recall@50": recall_at_k(ranked_groups, gold_groups, 50),
        "Recall@100": recall_at_k(ranked_groups, gold_groups, 100),
        "MRR@10": mrr_at_k(ranked_groups, gold_groups, 10),
        "NDCG@10": ndcg_at_k(ranked_groups, gold_groups, 10),
        "MAP@10": ap_at_k(ranked_groups, gold_groups, 10),
        "F2@2": f2_at_k(ranked_groups, gold_groups, 2),
        "F2@3": f2_at_k(ranked_groups, gold_groups, 3),
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Nhập và tính lại kết quả Task 9D/9E trên kho nhóm")
    parser.add_argument("--bm25-file", default=r"D:\VLQA\task9d\bm25_seg_top100_team.jsonl")
    parser.add_argument("--kaggle-dir", default=r"D:\VLQA\task9d\kaggle_out\team_eval_9d")
    parser.add_argument("--vlqa-file", default=str(BENCHMARK_DIR / "vlqa_test2190_team.jsonl"))
    parser.add_argument("--vlb-file", default=str(BENCHMARK_DIR / "vietlawbench_tune_team.jsonl"))
    parser.add_argument("--out-dir", default=str(BENCHMARK_DIR / "results"))
    parser.add_argument("--perquery-out", default=r"D:\VLQA\task9d\perquery_team_main.jsonl")
    parser.add_argument("--with-legacy", action="store_true", help="Đánh giá thêm bm25_legacy và system_sparse_legacy")
    parser.add_argument("--legacy-index", default="team_sparse_legacy_idx")
    parser.add_argument("--depth", type=int, default=100)
    args = parser.parse_args(argv)

    kaggle_dir = Path(args.kaggle_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    perquery_path = Path(args.perquery_out)
    perquery_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Nạp dữ liệu gold
    vlqa_items = load_jsonl(Path(args.vlqa_file))
    vlb_items = [r for r in load_jsonl(Path(args.vlb_file)) if r.get("status") == "mapped"]
    logger.info("Đã nạp %d câu hỏi VLQA và %d câu hỏi VietLawBench (mapped).", len(vlqa_items), len(vlb_items))

    # 2. Nạp danh sách xếp hạng BM25, Dense, Rerank
    bm25_map: Dict[Tuple[str, Any], List[str]] = {}
    for r in load_jsonl(Path(args.bm25_file)):
        bm25_map[(r["set"], r["qid"])] = r["ranked"]

    dense_map: Dict[Tuple[str, Any], List[str]] = {}
    for r in load_jsonl(kaggle_dir / "dense_top100_team.jsonl"):
        dense_map[(r["set"], r["qid"])] = r["ranked"]

    rerank_cands_map: Dict[Tuple[str, Any], List[str]] = {}
    rerank_ordered_map: Dict[Tuple[str, Any], List[str]] = {}
    for r in load_jsonl(kaggle_dir / "rerank_top50_team.jsonl"):
        key = (r["set"], r["qid"])
        rerank_cands_map[key] = r["candidates"]
        rerank_ordered_map[key] = order_rerank_candidates(r["candidates"], r["rerank_scores"])

    # 3. Tính cục bộ RRF k=60 và đối chiếu 50 ứng viên với Kaggle
    logger.info("Tính lại RRF k=60 cục bộ và kiểm tra đối chiếu top-50 candidates...")
    hybrid_map, mismatches = verify_rrf_candidates(
        bm25_map, dense_map, rerank_cands_map, k=60, depth=args.depth, top_n=50
    )
    logger.info("Số lượng câu hỏi không khớp RRF top-50 vs Kaggle candidates: %d / %d",
                len(mismatches), len(rerank_cands_map))
    if mismatches:
        logger.error("DỪNG: Phát hiện %d câu hỏi không khớp RRF top-50 (ví dụ: %s)", len(mismatches), mismatches[:5])
        sys.exit(1)

    # 4. Nếu --with-legacy: truy vấn thêm từ team_sparse_legacy_idx
    legacy_bm25_map: Dict[Tuple[str, Any], List[str]] = {}
    legacy_sys_map: Dict[Tuple[str, Any], List[str]] = {}
    if args.with_legacy:
        from elasticsearch import Elasticsearch
        from rag.es_retriever import LegalElasticsearchRetriever

        es_check = Elasticsearch([config.ES_HOST], request_timeout=60)
        if not es_check.indices.exists(index=args.legacy_index):
            logger.error("DỪNG: Chỉ mục %r chưa tồn tại! Hãy chạy pipeline.ingest_team_legacy trước.", args.legacy_index)
            sys.exit(1)

        logger.info("=== ĐÁNH GIÁ LEGACY ANALYZER TRÊN [%s] ===", args.legacy_index)
        retriever = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=args.legacy_index)
        es_client = retriever.client

        all_queries: List[Tuple[str, Any, str]] = []
        for item in vlqa_items:
            all_queries.append(("vlqa", item["qid"], item["query"]))
        for item in vlb_items:
            all_queries.append(("vlb", item["benchmark_id"], item["query"]))

        lat_bm25_legacy: List[float] = []
        lat_sys_legacy: List[float] = []

        for idx, (qset, qid, query_text) in enumerate(all_queries):
            key = (qset, qid)

            t0 = time.perf_counter()
            resp = es_client.search(
                index=args.legacy_index,
                query={"match": {"content": query_text}},
                size=args.depth,
                _source=["chunk_id"],
            )
            lat_bm25_legacy.append((time.perf_counter() - t0) * 1000.0)
            raw_bm25 = [h["_source"]["chunk_id"] for h in resp["hits"]["hits"]]
            legacy_bm25_map[key] = deduplicate_keep_order(raw_bm25)[:args.depth]

            t1 = time.perf_counter()
            hits = retriever.search_sparse(query=query_text, top_k=args.depth, must_be_effective=True)
            lat_sys_legacy.append((time.perf_counter() - t1) * 1000.0)
            raw_sys = [str(h.get("chunk_id") or h.get("id")) for h in hits]
            legacy_sys_map[key] = deduplicate_keep_order(raw_sys)[:args.depth]

            if (idx + 1) % 500 == 0 or (idx + 1) == len(all_queries):
                logger.info("  Legacy search: %d/%d câu hỏi...", idx + 1, len(all_queries))

        logger.info(
            "Độ trễ legacy (ms): bm25_legacy P50=%.2f P95=%.2f | system_sparse_legacy P50=%.2f P95=%.2f",
            float(np.percentile(lat_bm25_legacy, 50)),
            float(np.percentile(lat_bm25_legacy, 95)),
            float(np.percentile(lat_sys_legacy, 50)),
            float(np.percentile(lat_sys_legacy, 95)),
        )

    # 5. Tính điểm tất cả chế độ
    base_modes = ["bm25_seg", "dense", "hybrid_rrf60", "rerank_top50"]
    all_modes = list(base_modes)
    if args.with_legacy:
        all_modes.extend(["bm25_legacy", "system_sparse_legacy"])

    subset_names = ["vlqa_all_2190", "vlqa_reachable_full", "vlqa_dev500_all", "vlb_mapped_710"]
    results: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        m: {s: [] for s in subset_names} for m in all_modes
    }
    vlqa_per_qid: Dict[str, Dict[int, Dict[str, float]]] = {m: {} for m in all_modes}
    vlb_per_qid: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in all_modes}

    def get_mode_ranking(mode: str, key: Tuple[str, Any]) -> List[str]:
        if mode == "bm25_seg":
            return bm25_map[key]
        if mode == "dense":
            return dense_map[key]
        if mode == "hybrid_rrf60":
            return hybrid_map[key]
        if mode == "rerank_top50":
            return rerank_ordered_map[key]
        if mode == "bm25_legacy":
            return legacy_bm25_map[key]
        if mode == "system_sparse_legacy":
            return legacy_sys_map[key]
        raise ValueError(f"Unknown mode: {mode}")

    with open(perquery_path, "w", encoding="utf-8") as pq_f:
        for item in vlqa_items:
            qid = item["qid"]
            key = ("vlqa", qid)
            chunk_to_group: Dict[str, str] = {}
            gold_groups: Set[str] = set()
            for aid_str, ainfo in item.get("per_aid", {}).items():
                gid = f"aid:{aid_str}"
                gold_groups.add(gid)
                for cid in ainfo.get("gold_chunks", []):
                    chunk_to_group[cid] = gid

            for mode in all_modes:
                ranked_cids = get_mode_ranking(mode, key)
                m = evaluate_ranking(ranked_cids, chunk_to_group, gold_groups)
                results[mode]["vlqa_all_2190"].append(m)
                if item.get("status") == "full":
                    results[mode]["vlqa_reachable_full"].append(m)
                if item.get("orig_split") == "dev":
                    results[mode]["vlqa_dev500_all"].append(m)
                vlqa_per_qid[mode][qid] = m

                rec = {"set": "vlqa", "qid": qid, "mode": mode}
                for mk in METRIC_KEYS:
                    rec[mk] = round(m[mk], 6)
                pq_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for item in vlb_items:
            bid = item["benchmark_id"]
            key = ("vlb", bid)
            gid = f"vlb:{bid}"
            chunk_to_group = {cid: gid for cid in item.get("relevant_chunk_ids", [])}
            gold_groups = {gid}

            for mode in all_modes:
                ranked_cids = get_mode_ranking(mode, key)
                m = evaluate_ranking(ranked_cids, chunk_to_group, gold_groups)
                results[mode]["vlb_mapped_710"].append(m)
                vlb_per_qid[mode][bid] = m

                rec = {"set": "vlb", "qid": bid, "mode": mode}
                for mk in METRIC_KEYS:
                    rec[mk] = round(m[mk], 6)
                pq_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("✓ Đã lưu kết quả từng câu hỏi vào: %s", perquery_path)

    # 6. Đối chiếu với team_dense_rerank_metrics.csv của Kaggle (chính xác <= 0.0005)
    kaggle_metrics_path = kaggle_dir / "team_dense_rerank_metrics.csv"
    kaggle_lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(kaggle_metrics_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kaggle_lookup[(row["mode"], row["subset"])] = row

    metric_rows: List[Dict[str, Any]] = []
    max_diff = 0.0
    for subset in subset_names:
        for mode in all_modes:
            q_list = results[mode][subset]
            agg = aggregate(q_list)
            row: Dict[str, Any] = {
                "mode": mode,
                "subset": subset,
                "n_queries": len(q_list),
            }
            for mk in METRIC_KEYS:
                val_4 = round(agg[mk], 4)
                row[mk] = val_4
                if mode in base_modes:
                    k_val = float(kaggle_lookup[(mode, subset)][mk])
                    diff = abs(val_4 - k_val)
                    if diff > max_diff:
                        max_diff = diff
                    if diff > 0.0005:
                        logger.error(
                            "DỪNG: Sai lệch vượt 0.0005 ở (%s, %s, %s): local=%.4f vs kaggle=%.4f",
                            mode, subset, mk, val_4, k_val,
                        )
                        sys.exit(1)
            metric_rows.append(row)

    logger.info("✓ Đối chiếu toàn bộ 4 chế độ x 4 tập con khớp Kaggle (sai lệch tối đa = %.6f <= 0.0005)", max_diff)

    # Ghi team_main_metrics.csv
    metrics_csv_path = out_dir / "team_main_metrics.csv"
    metrics_headers = ["mode", "subset", "n_queries"] + METRIC_KEYS
    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric_rows)
    logger.info("✓ Đã lưu bảng chỉ số vào: %s", metrics_csv_path)

    # 7. Kiểm định ý nghĩa thống kê ghép cặp (paired t-test)
    ttest_pairs: List[Tuple[str, str]] = [
        ("dense", "bm25_seg"),
        ("hybrid_rrf60", "dense"),
        ("hybrid_rrf60", "bm25_seg"),
        ("rerank_top50", "dense"),
        ("rerank_top50", "hybrid_rrf60"),
    ]
    if args.with_legacy:
        ttest_pairs.extend([
            ("bm25_legacy", "bm25_seg"),
            ("system_sparse_legacy", "bm25_seg"),
        ])

    vlqa_qids = sorted(vlqa_per_qid["bm25_seg"].keys())
    vlb_bids = sorted(vlb_per_qid["bm25_seg"].keys())

    sig_rows: List[Dict[str, Any]] = []
    for subset_label, per_qid_dict, qid_list in [
        ("vlqa_all_2190", vlqa_per_qid, vlqa_qids),
        ("vlb_mapped_710", vlb_per_qid, vlb_bids),
    ]:
        for mode_name, base_name in ttest_pairs:
            for met in ["NDCG@10", "F2@2"]:
                m_vals = [per_qid_dict[mode_name][q][met] for q in qid_list]
                b_vals = [per_qid_dict[base_name][q][met] for q in qid_list]
                t_res = run_paired_ttest(
                    m_vals, b_vals,
                    dataset_name=subset_label,
                    mode_name=mode_name,
                    base_name=base_name,
                    metric_name=met,
                )
                sig_rows.append({
                    "subset": t_res["dataset"],
                    "mode": t_res["mode"],
                    "baseline": t_res["baseline"],
                    "metric": t_res["metric"],
                    "mean_mode": t_res["mean_mode"],
                    "mean_baseline": t_res["mean_baseline"],
                    "mean_diff": t_res["mean_diff"],
                    "t_stat": t_res["t_stat"],
                    "p_value": t_res["p_value"],
                    "n": t_res["n"],
                })

    sig_csv_path = out_dir / "team_main_significance.csv"
    sig_headers = ["subset", "mode", "baseline", "metric", "mean_mode", "mean_baseline", "mean_diff", "t_stat", "p_value", "n"]
    with open(sig_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sig_headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sig_rows)
    logger.info("✓ Đã lưu bảng kiểm định ý nghĩa thống kê vào: %s", sig_csv_path)

    # 8. Lưu team_gpu_latency.json
    gpu_lat_src = kaggle_dir / "gpu_latency.json"
    gpu_lat_data = json.loads(gpu_lat_src.read_text(encoding="utf-8"))
    gpu_lat_data["note"] = "GPU = Kaggle T4; BM25 not included (laptop ES, P50 ≈ 42 ms in Task 9C)"
    gpu_lat_dst = out_dir / "team_gpu_latency.json"
    with open(gpu_lat_dst, "w", encoding="utf-8") as f:
        json.dump(gpu_lat_data, f, ensure_ascii=False, indent=2)
    logger.info("✓ Đã lưu độ trễ GPU vào: %s", gpu_lat_dst)

    print("\n" + "=" * 95)
    print(f"HOÀN THÀNH import_team_eval_9d (--with-legacy={args.with_legacy}) trong {time.perf_counter() - t_start:.2f} giây")
    print("=" * 95)


if __name__ == "__main__":
    main()
