r"""
benchmark/rerank_topn.py — Đánh giá chất lượng ngoại tuyến khi chỉ rerank top-N ứng viên.

Quy tắc:
- Chọn ngưỡng N trên tập tune1690 (N nhỏ nhất thỏa mãn NDCG@10 >= NDCG@10(N=50) - tol).
- Sau khi có N tối ưu từ tune, báo cáo kết quả tương ứng trên tập dev500.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger
from benchmark.metrics_article import ndcg_at_k, recall_at_k, mrr_at_k, f2_at_k

logger = get_subsystem_logger("benchmark", "rerank_topn")


def rerank_first_n(candidates: List[str], scores: List[float], n: int) -> List[str]:
    """
    Sắp xếp lại n ứng viên đầu tiên theo rerank_scores giảm dần (hòa điểm: giữ thứ tự gốc).
    Các ứng viên từ vị trí n trở đi giữ nguyên thứ tự ban đầu.
    n <= 0 trả về danh sách candidates gốc không đổi.
    n >= len(candidates) sắp xếp lại toàn bộ candidates.
    """
    if n <= 0:
        return list(candidates)
    if n >= len(candidates):
        top_c = candidates
        top_s = scores
        sorted_top = [c for c, _ in sorted(zip(top_c, top_s), key=lambda x: x[1], reverse=True)]
        return sorted_top

    top_c = candidates[:n]
    top_s = scores[:n]
    sorted_top = [c for c, _ in sorted(zip(top_c, top_s), key=lambda x: x[1], reverse=True)]
    return sorted_top + list(candidates[n:])


def choose_n(rows: List[Dict[str, Any]], tol: float = 0.005) -> int:
    """
    Chọn n nhỏ nhất trên tập tune thỏa mãn:
    NDCG@10(n) >= NDCG@10(n=50) - tol.
    """
    target_row = next((r for r in rows if r["n"] == 50), None)
    if target_row is None:
        raise ValueError("Không tìm thấy hàng n=50 trong danh sách kết quả tune")
    target_ndcg = float(target_row["NDCG@10"])
    eligible = [r for r in rows if float(r["NDCG@10"]) >= target_ndcg - tol]
    if not eligible:
        return 50
    return min(int(r["n"]) for r in eligible)


def load_gold_map(path: Path | str) -> Dict[int, Set[str]]:
    gold_map: Dict[int, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_map[int(item["qid"])] = set(item.get("relevant_chunk_ids", []))
    return gold_map


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    raw_vlqa_dir = os.environ.get("VLQA_DIR") or getattr(config, "VLQA_DIR", "")
    vlqa_dir = raw_vlqa_dir if raw_vlqa_dir else "D:\\VLQA"
    default_rerank_file = str(Path(vlqa_dir) / "task6" / "kaggle_out" / "rerank_top50.jsonl")

    ap = argparse.ArgumentParser(description="Đánh giá chất lượng khi rerank top-N ứng viên")
    ap.add_argument("--rerank-file", default=default_rerank_file)
    ap.add_argument("--dev-file", default=str(BENCHMARK_DIR / "vlqa_dev500.jsonl"))
    ap.add_argument("--tune-file", default=str(BENCHMARK_DIR / "vlqa_tune1690.jsonl"))
    ap.add_argument("--out", default=str(BENCHMARK_DIR / "results" / "vlqa_rerank_topn_quality.csv"))
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args(argv)

    rerank_path = Path(args.rerank_file)
    if not rerank_path.exists():
        logger.error("Không tìm thấy file rerank: %s", rerank_path)
        sys.exit(1)

    logger.info("Nạp tập nhãn vàng dev và tune...")
    gold_tune = load_gold_map(args.tune_file)
    gold_dev = load_gold_map(args.dev_file)

    records_tune: List[Dict[str, Any]] = []
    records_dev: List[Dict[str, Any]] = []

    logger.info("Nạp các bản ghi rerank từ %s...", rerank_path)
    with open(rerank_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sp = item.get("split")
                if sp == "tune":
                    records_tune.append(item)
                elif sp == "dev":
                    records_dev.append(item)

    logger.info("Đã nạp: %d bản ghi tune, %d bản ghi dev", len(records_tune), len(records_dev))

    n_values = [0, 5, 10, 20, 30, 50]

    # 1. Đánh giá trên tập tune
    tune_metrics: List[Dict[str, Any]] = []
    for n in n_values:
        ndcg_l, rec_l, mrr_l, f2_l = [], [], [], []
        for r in records_tune:
            ranked = rerank_first_n(r["candidates"], r["rerank_scores"], n)
            g = gold_tune.get(int(r["qid"]), set())
            ndcg_l.append(ndcg_at_k(ranked, g, 10))
            rec_l.append(recall_at_k(ranked, g, 10))
            mrr_l.append(mrr_at_k(ranked, g, 10))
            f2_l.append(f2_at_k(ranked, g, 2))

        row = {
            "split": "tune",
            "n": n,
            "NDCG@10": sum(ndcg_l) / len(ndcg_l),
            "Recall@10": sum(rec_l) / len(rec_l),
            "MRR@10": sum(mrr_l) / len(mrr_l),
            "F2@2": sum(f2_l) / len(f2_l),
        }
        tune_metrics.append(row)

    # Chọn N trên tune
    chosen_n = choose_n(tune_metrics, tol=args.tol)
    logger.info("✓ N tối ưu được chọn trên tập tune (tol=%.4f): N=%d", args.tol, chosen_n)

    # 2. Đánh giá trên tập dev
    dev_metrics: List[Dict[str, Any]] = []
    for n in n_values:
        ndcg_l, rec_l, mrr_l, f2_l = [], [], [], []
        for r in records_dev:
            ranked = rerank_first_n(r["candidates"], r["rerank_scores"], n)
            g = gold_dev.get(int(r["qid"]), set())
            ndcg_l.append(ndcg_at_k(ranked, g, 10))
            rec_l.append(recall_at_k(ranked, g, 10))
            mrr_l.append(mrr_at_k(ranked, g, 10))
            f2_l.append(f2_at_k(ranked, g, 2))

        row = {
            "split": "dev",
            "n": n,
            "NDCG@10": sum(ndcg_l) / len(ndcg_l),
            "Recall@10": sum(rec_l) / len(rec_l),
            "MRR@10": sum(mrr_l) / len(mrr_l),
            "F2@2": sum(f2_l) / len(f2_l),
        }
        dev_metrics.append(row)

    # Ghi file CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["split", "n", "NDCG@10", "Recall@10", "MRR@10", "F2@2", "chosen_on_tune"]

    all_rows = []
    for r in tune_metrics:
        all_rows.append({
            "split": "tune",
            "n": r["n"],
            "NDCG@10": f"{r['NDCG@10']:.4f}",
            "Recall@10": f"{r['Recall@10']:.4f}",
            "MRR@10": f"{r['MRR@10']:.4f}",
            "F2@2": f"{r['F2@2']:.4f}",
            "chosen_on_tune": "True" if r["n"] == chosen_n else "False",
        })

    for r in dev_metrics:
        all_rows.append({
            "split": "dev",
            "n": r["n"],
            "NDCG@10": f"{r['NDCG@10']:.4f}",
            "Recall@10": f"{r['Recall@10']:.4f}",
            "MRR@10": f"{r['MRR@10']:.4f}",
            "F2@2": f"{r['F2@2']:.4f}",
            "chosen_on_tune": "True" if r["n"] == chosen_n else "False",
        })

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("✓ Đã lưu bảng chất lượng top-N rerank vào %s", out_path)

    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ RERANK TOP-N (TUNE VÀ DEV):")
    print("=" * 80)
    with open(out_path, "r", encoding="utf-8") as f:
        print(f.read().strip())
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()