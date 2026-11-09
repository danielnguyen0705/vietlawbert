r"""
benchmark/tune_vlqa_fusion.py — Chọn trọng số RRF và ngưỡng cắt reranker trên tập tune 1.690 câu.

Quy tắc nghiêm ngặt:
- TUYỆT ĐỐI KHÔNG đọc bất kỳ dữ liệu nào của tập dev trong script này.
- Mọi bản ghi nạp vào phải có split == "tune".
- Tối ưu tham số trên tune1690, sau đó tập dev chỉ được đánh giá 1 lần duy nhất.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger
from benchmark.metrics_article import ndcg_at_k, recall_at_k, f2_at_k

logger = get_subsystem_logger("benchmark", "tune_vlqa_fusion")


def weighted_rrf(
    rankings: List[List[str]],
    weights: List[float],
    k: int,
    depth: int = 100,
) -> List[str]:
    """
    Weighted Reciprocal Rank Fusion:
    score(c) = sum_i weights[i] / (k + rank_i(c)), với rank_i bắt đầu từ 1.
    Sắp xếp theo score giảm dần, hòa điểm (tie) xếp theo chunk_id tăng dần.
    Trả về top depth chunk_id.
    """
    scores: Dict[str, float] = {}
    for r_list, w in zip(rankings, weights):
        for rank, cid in enumerate(r_list, start=1):
            if cid:
                scores[cid] = scores.get(cid, 0.0) + float(w) / (k + rank)

    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [cid for cid, _score in sorted_items[:depth]]


def order_rerank_candidates(candidates: List[str], scores: List[float]) -> List[str]:
    """
    Sắp xếp các ứng viên theo rerank_score giảm dần.
    Python sort là stable sort nên khi hòa điểm (tie) thứ tự gốc của candidate được bảo toàn.
    """
    sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _score in sorted_pairs]


def load_tune_records(records_or_path: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Nạp và kiểm định các bản ghi chỉ thuộc tập tune.
    Ném lỗi ValueError nếu có bất kỳ bản ghi nào có split == 'dev' hoặc split != 'tune'.
    """
    if isinstance(records_or_path, (str, Path)):
        records: List[Dict[str, Any]] = []
        with open(records_or_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        records = list(records_or_path)

    for r in records:
        split_val = r.get("split")
        if split_val == "dev":
            raise ValueError(f"CẤM: Phát hiện bản ghi dev (qid={r.get('qid')}) trong quy trình tuning!")
        if split_val != "tune":
            raise ValueError(f"Bản ghi phải có split == 'tune', nhận được: '{split_val}' (qid={r.get('qid')})")

    return records


def extract_tune_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Trích xuất các bản ghi thuộc split == 'tune' từ file jsonl hỗn hợp, sau đó kiểm định hợp lệ.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("split") == "tune":
                    records.append(item)

    if not records:
        raise ValueError(f"Không tìm thấy bản ghi tune nào trong {path}")

    return load_tune_records(records)


def select_best_fusion_params(grid_rows: List[Dict[str, Any]], ndcg_tol: float = 0.0005) -> Dict[str, Any]:
    """
    Quy tắc chọn tham số tối ưu trên tune:
    1. NDCG@10 cao nhất; các ứng viên có chênh lệch < 0.0005 so với max_ndcg được coi là hòa điểm.
    2. Nếu hòa NDCG: chọn F2@2 cao nhất.
    3. Nếu hòa F2@2: chọn k nhỏ hơn.
    4. Nếu hòa k: chọn w_dense nhỏ hơn.
    """
    if not grid_rows:
        raise ValueError("Danh sách grid_rows rỗng!")

    max_ndcg = max(r["NDCG@10"] for r in grid_rows)
    candidates = [r for r in grid_rows if (max_ndcg - r["NDCG@10"]) < ndcg_tol]

    max_f2 = max(r["F2@2"] for r in candidates)
    candidates = [r for r in candidates if abs(r["F2@2"] - max_f2) < 1e-9]

    min_k = min(r["k"] for r in candidates)
    candidates = [r for r in candidates if r["k"] == min_k]

    min_w = min(r["w_dense"] for r in candidates)
    candidates = [r for r in candidates if abs(r["w_dense"] - min_w) < 1e-9]

    return candidates[0]


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    vlqa_dir_default = os.environ.get("VLQA_DIR") or getattr(config, "VLQA_DIR", "")
    default_base = vlqa_dir_default if vlqa_dir_default else "D:\\VLQA"

    default_bm25_file = str(Path(default_base) / "task6" / "bm25_seg_top100.jsonl")
    default_dense_file = str(Path(default_base) / "task6" / "kaggle_out" / "dense_top100.jsonl")
    default_rerank_file = str(Path(default_base) / "task6" / "kaggle_out" / "rerank_top50.jsonl")
    default_tune_file = str(BENCHMARK_DIR / "vlqa_tune1690.jsonl")
    default_grid_csv = str(BENCHMARK_DIR / "results" / "vlqa_tune1690_fusion_grid.csv")
    default_params_json = str(BENCHMARK_DIR / "results" / "vlqa_fusion_params.json")

    ap = argparse.ArgumentParser(description="Tuning tham số Weighted RRF và ngưỡng reranker trên tập tune")
    ap.add_argument("--bm25-file", default=default_bm25_file)
    ap.add_argument("--dense-file", default=default_dense_file)
    ap.add_argument("--rerank-file", default=default_rerank_file)
    ap.add_argument("--tune-file", default=default_tune_file)
    ap.add_argument("--out-grid", default=default_grid_csv)
    ap.add_argument("--out-params", default=default_params_json)
    args = ap.parse_args(argv)

    # 1. Nạp và xác thực dữ liệu TUNE
    logger.info("Nạp dữ liệu xếp hạng BM25 (tune) từ %s...", args.bm25_file)
    bm25_tune = extract_tune_records(args.bm25_file)

    logger.info("Nạp dữ liệu xếp hạng Dense (tune) từ %s...", args.dense_file)
    dense_tune = extract_tune_records(args.dense_file)

    logger.info("Nạp dữ liệu Rerank (tune) từ %s...", args.rerank_file)
    rerank_tune = extract_tune_records(args.rerank_file)

    logger.info("Nạp ground truth tập tune từ %s...", args.tune_file)
    tune_gold_cases = []
    with open(args.tune_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tune_gold_cases.append(json.loads(line))

    assert len(tune_gold_cases) == 1690, f"Kỳ vọng 1690 câu tune, nhận được {len(tune_gold_cases)}"
    assert len(bm25_tune) == 1690, f"Lệch số câu bm25_tune: {len(bm25_tune)}"
    assert len(dense_tune) == 1690, f"Lệch số câu dense_tune: {len(dense_tune)}"
    assert len(rerank_tune) == 1690, f"Lệch số câu rerank_tune: {len(rerank_tune)}"

    gold_map = {q["qid"]: set(q.get("relevant_chunk_ids", [])) for q in tune_gold_cases}

    # Đảm bảo khớp thứ tự qid
    bm25_qids = [r["qid"] for r in bm25_tune]
    dense_qids = [r["qid"] for r in dense_tune]
    rerank_qids = [r["qid"] for r in rerank_tune]
    tune_qids = [q["qid"] for q in tune_gold_cases]

    assert bm25_qids == tune_qids, "QID bm25 không khớp tune gold"
    assert dense_qids == tune_qids, "QID dense không khớp tune gold"
    assert rerank_qids == tune_qids, "QID rerank không khớp tune gold"

    # 2. Grid Search Weighted RRF
    w_dense_vals = [0.5, 0.6, 0.7, 0.8, 0.9]
    k_vals = [10, 20, 30, 60, 100]

    logger.info("Bắt đầu duyệt lưới Weighted RRF (5 weights x 5 k = 25 tổ hợp) trên 1.690 câu...")
    grid_rows: List[Dict[str, Any]] = []

    for wd in w_dense_vals:
        wb = round(1.0 - wd, 4)
        weights = [wb, wd]
        for k in k_vals:
            ndcg_list = []
            r10_list = []
            r100_list = []
            f2_2_list = []
            f2_3_list = []

            for b_rec, d_rec in zip(bm25_tune, dense_tune):
                qid = b_rec["qid"]
                golds = gold_map[qid]
                ranked = weighted_rrf([b_rec["ranked"], d_rec["ranked"]], weights, k, depth=100)
                ndcg_list.append(ndcg_at_k(ranked, golds, 10))
                r10_list.append(recall_at_k(ranked, golds, 10))
                r100_list.append(recall_at_k(ranked, golds, 100))
                f2_2_list.append(f2_at_k(ranked, golds, 2))
                f2_3_list.append(f2_at_k(ranked, golds, 3))

            row = {
                "w_dense": wd,
                "w_bm25": wb,
                "k": k,
                "NDCG@10": sum(ndcg_list) / len(ndcg_list),
                "Recall@10": sum(r10_list) / len(r10_list),
                "Recall@100": sum(r100_list) / len(r100_list),
                "F2@2": sum(f2_2_list) / len(f2_2_list),
                "F2@3": sum(f2_3_list) / len(f2_3_list),
            }
            grid_rows.append(row)

    # 3. Chọn tham số tốt nhất
    best_fusion = select_best_fusion_params(grid_rows)
    logger.info(
        "✓ Tham số Weighted RRF tối ưu: w_dense=%.2f, w_bm25=%.2f, k=%d (NDCG@10=%.4f, F2@2=%.4f)",
        best_fusion["w_dense"],
        best_fusion["w_bm25"],
        best_fusion["k"],
        best_fusion["NDCG@10"],
        best_fusion["F2@2"],
    )

    # 4. Xác định ngưỡng cắt k tối ưu cho Reranker
    logger.info("Đang đánh giá ngưỡng cắt k cho bge-reranker-v2-m3 trên tập tune...")
    rerank_ordered_by_qid = {}
    for r in rerank_tune:
        qid = r["qid"]
        ordered = order_rerank_candidates(r["candidates"], r["rerank_scores"])
        rerank_ordered_by_qid[qid] = ordered

    rerank_f2_scores: Dict[int, float] = {}
    for k_cut in [1, 2, 3, 5]:
        f2_scores = [
            f2_at_k(rerank_ordered_by_qid[qid], gold_map[qid], k_cut)
            for qid in tune_qids
        ]
        mean_f2 = sum(f2_scores) / len(f2_scores)
        rerank_f2_scores[k_cut] = mean_f2
        logger.info("  Rerank tune Macro-F2@%d = %.4f", k_cut, mean_f2)

    # Chọn k có F2 cao nhất; nếu hòa điểm chọn k nhỏ hơn
    best_rerank_k = 1
    best_rerank_f2 = -1.0
    for k_cut in [1, 2, 3, 5]:
        val = rerank_f2_scores[k_cut]
        if val > best_rerank_f2:
            best_rerank_f2 = val
            best_rerank_k = k_cut

    logger.info("✓ Ngưỡng cắt reranker tối ưu: k=%d (Macro-F2=%.4f)", best_rerank_k, best_rerank_f2)

    # 5. Lưu bảng kết quả grid CSV
    out_grid_path = Path(args.out_grid)
    out_grid_path.parent.mkdir(parents=True, exist_ok=True)
    csv_headers = ["w_dense", "w_bm25", "k", "NDCG@10", "Recall@10", "Recall@100", "F2@2", "F2@3"]

    with open(out_grid_path, "w", encoding="utf-8", newline="") as gf:
        writer = csv.DictWriter(gf, fieldnames=csv_headers, lineterminator="\n")
        writer.writeheader()
        for r in grid_rows:
            formatted_row = {
                "w_dense": f"{r['w_dense']:.4f}",
                "w_bm25": f"{r['w_bm25']:.4f}",
                "k": r["k"],
                "NDCG@10": f"{r['NDCG@10']:.4f}",
                "Recall@10": f"{r['Recall@10']:.4f}",
                "Recall@100": f"{r['Recall@100']:.4f}",
                "F2@2": f"{r['F2@2']:.4f}",
                "F2@3": f"{r['F2@3']:.4f}",
            }
            writer.writerow(formatted_row)

    logger.info("✓ Đã lưu bảng duyệt lưới tune vào %s", out_grid_path)

    # 6. Lưu tham số tối ưu vào file JSON
    out_params_path = Path(args.out_params)
    out_params_path.parent.mkdir(parents=True, exist_ok=True)

    params_payload = {
        "w_dense": best_fusion["w_dense"],
        "w_bm25": best_fusion["w_bm25"],
        "rrf_k": best_fusion["k"],
        "rerank_best_f2_k": best_rerank_k,
        "chosen_on": "tune1690",
        "tune_NDCG@10": round(best_fusion["NDCG@10"], 4),
        "tune_Recall@10": round(best_fusion["Recall@10"], 4),
        "tune_Recall@100": round(best_fusion["Recall@100"], 4),
        "tune_F2@2": round(best_fusion["F2@2"], 4),
        "tune_F2@3": round(best_fusion["F2@3"], 4),
        "rerank_tune_F2_at_best_k": round(best_rerank_f2, 4),
    }

    with open(out_params_path, "w", encoding="utf-8") as pf:
        json.dump(params_payload, pf, indent=2, ensure_ascii=False)
        pf.write("\n")

    logger.info("✓ Đã lưu file cấu hình tham số tối ưu vào %s", out_params_path)

    # In báo cáo console
    print("\n" + "=" * 80)
    print("KẾT QUẢ TỐI ƯU HÓA FUSION VÀ RERANKER TRÊN TẬP TUNE 1.690 CÂU:")
    print("=" * 80)
    print(f"Top 1 Weighted RRF: w_dense={best_fusion['w_dense']}, w_bm25={best_fusion['w_bm25']}, k={best_fusion['k']}")
    print(f"  NDCG@10    = {best_fusion['NDCG@10']:.4f}")
    print(f"  Recall@10  = {best_fusion['Recall@10']:.4f}")
    print(f"  Recall@100 = {best_fusion['Recall@100']:.4f}")
    print(f"  F2@2       = {best_fusion['F2@2']:.4f}")
    print(f"  F2@3       = {best_fusion['F2@3']:.4f}")
    print("-" * 80)
    print(f"Reranker bge-reranker-v2-m3:")
    print(f"  Best F2@k trên tune: k={best_rerank_k} (Macro-F2 = {best_rerank_f2:.4f})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()