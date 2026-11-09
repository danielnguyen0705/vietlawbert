"""
benchmark/export_vlqa_candidates.py — Xuất danh sách ứng viên BM25 tách từ (top-100) cho dev và tune.

Mục đích:
- Tạo candidates top-100 từ BM25 có tách từ (pyvi) cho 500 câu dev và 1.690 câu tune.
- Xuất file vào thư mục ngoài repo (ví dụ D:\VLQA\task6\) để tải lên Kaggle phục vụ bước Reranking.
- Sao chép byte-identical file tập dev và tune vào thư mục xuất.
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger
from preprocess.vi_segment import get_segmenter, segment_text
from benchmark.metrics_article import recall_at_k, ndcg_at_k

logger = get_subsystem_logger("benchmark", "export_vlqa_candidates")


def is_inside_repo(path: Path | str, repo_root: Path = ROOT_DIR) -> bool:
    """Kiểm tra đường dẫn có nằm bên trong hoặc trùng với thư mục repository hay không."""
    try:
        resolved_path = Path(path).resolve()
        resolved_root = Path(repo_root).resolve()
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def deduplicate_hits(hits: List[Tuple[str, float]], depth: int = 100) -> Tuple[List[str], List[float]]:
    """Loại bỏ trùng lặp giữ lần xuất hiện đầu tiên (điểm cao nhất), giới hạn tối đa depth phần tử."""
    seen = set()
    ranked_ids: List[str] = []
    ranked_scores: List[float] = []
    for cid, score in hits:
        if cid and cid not in seen:
            seen.add(cid)
            ranked_ids.append(cid)
            ranked_scores.append(float(score))
            if depth > 0 and len(ranked_ids) >= depth:
                break
    return ranked_ids, ranked_scores


def export_rankings(
    questions: List[Dict[str, Any]],
    split: str,
    search_fn: Callable[[str], List[Tuple[str, float]]],
    depth: int = 100,
) -> List[Dict[str, Any]]:
    """
    Hàm thuần túy sinh danh sách xếp hạng cho tập câu hỏi:
    {"split": split, "qid": <int>, "ranked": [chunk ids], "scores": [ES scores]}
    """
    results: List[Dict[str, Any]] = []
    for q in questions:
        raw_qid = q.get("qid")
        try:
            qid_val = int(raw_qid)
        except (ValueError, TypeError):
            qid_val = raw_qid

        query_text = q.get("query", "")
        raw_hits = search_fn(query_text)
        ranked_ids, ranked_scores = deduplicate_hits(raw_hits, depth=depth)

        results.append({
            "split": split,
            "qid": qid_val,
            "ranked": ranked_ids,
            "scores": ranked_scores,
        })
    return results


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    vlqa_dir_default = os.environ.get("VLQA_DIR") or getattr(config, "VLQA_DIR", "")
    default_base = vlqa_dir_default if vlqa_dir_default else "D:\\VLQA"
    default_out_dir = str(Path(default_base) / "task6")

    ap = argparse.ArgumentParser(description="Xuất danh sách ứng viên BM25 tách từ cho dev và tune")
    ap.add_argument("--dev", default=str(BENCHMARK_DIR / "vlqa_dev500.jsonl"))
    ap.add_argument("--tune", default=str(BENCHMARK_DIR / "vlqa_tune1690.jsonl"))
    ap.add_argument("--seg-index", default="vlqa_sparse_seg_idx")
    ap.add_argument("--segmenter", default="pyvi")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--out-dir", default=default_out_dir)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    # 1. Guards
    main_es_idx = getattr(config, "ES_INDEX_NAME", "vietlaw_sparse_idx")
    if args.seg_index == main_es_idx:
        logger.error("CẤM: --seg-index trùng với config.ES_INDEX_NAME ('%s').", args.seg_index)
        sys.exit(2)

    if is_inside_repo(args.out_dir):
        logger.error("CẤM: --out-dir không được nằm trong repository (%s).", args.out_dir)
        sys.exit(2)

    dev_path = Path(args.dev)
    tune_path = Path(args.tune)
    if not dev_path.exists():
        logger.error("Không tìm thấy file dev: %s", dev_path)
        sys.exit(1)
    if not tune_path.exists():
        logger.error("Không tìm thấy file tune: %s", tune_path)
        sys.exit(1)

    dev_questions: List[Dict[str, Any]] = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_questions.append(json.loads(line))

    tune_questions: List[Dict[str, Any]] = []
    with open(tune_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tune_questions.append(json.loads(line))

    if args.limit > 0:
        dev_questions = dev_questions[: args.limit]
        tune_questions = tune_questions[: args.limit]

    logger.info("Nạp %d câu hỏi dev từ %s", len(dev_questions), dev_path)
    logger.info("Nạp %d câu hỏi tune từ %s", len(tune_questions), tune_path)

    # 2. Khởi tạo segmenter và Elasticsearch
    seg_fn = get_segmenter(args.segmenter)

    from elasticsearch import Elasticsearch
    es_client = Elasticsearch(config.ES_HOST, timeout=60.0)

    def es_search_fn(query_text: str) -> List[Tuple[str, float]]:
        seg_query = segment_text(query_text, seg_fn)
        body = {
            "query": {"match": {"content_seg": seg_query}},
            "size": args.depth,
            "_source": ["chunk_id"],
        }
        resp = es_client.search(index=args.seg_index, **body)
        return [(hit["_id"], float(hit["_score"])) for hit in resp["hits"]["hits"]]

    logger.info("Đang truy xuất ứng viên cho tập dev (%d câu)...", len(dev_questions))
    dev_records = export_rankings(dev_questions, "dev", es_search_fn, depth=args.depth)

    logger.info("Đang truy xuất ứng viên cho tập tune (%d câu)...", len(tune_questions))
    tune_records = export_rankings(tune_questions, "tune", es_search_fn, depth=args.depth)

    # 3. Cross-check tập dev với kết quả baseline
    if len(dev_questions) == 500 and not args.limit:
        r10_list = []
        ndcg10_list = []
        for q, rec in zip(dev_questions, dev_records):
            gold = set(q.get("relevant_chunk_ids", []))
            r10_list.append(recall_at_k(rec["ranked"], gold, 10))
            ndcg10_list.append(ndcg_at_k(rec["ranked"], gold, 10))

        mean_r10 = sum(r10_list) / len(r10_list)
        mean_ndcg10 = sum(ndcg10_list) / len(ndcg10_list)

        print("\n" + "=" * 80)
        print("KIỂM TRA CHÉO (CROSS-CHECK) KẾT QUẢ DEV BM25-SEG VS BASELINE METRICS:")
        print(f"  Recall@10: thực tế = {mean_r10:.4f}, kỳ vọng = 0.6666 (lệch = {abs(mean_r10 - 0.6666):.4f})")
        print(f"  NDCG@10  : thực tế = {mean_ndcg10:.4f}, kỳ vọng = 0.5161 (lệch = {abs(mean_ndcg10 - 0.5161):.4f})")
        print("=" * 80 + "\n")

        if abs(mean_r10 - 0.6666) > 0.001 or abs(mean_ndcg10 - 0.5161) > 0.001:
            logger.error("LỖI: Kết quả dev bm25-seg không khớp với benchmark baseline trong ngưỡng 0.001!")
            sys.exit(1)
        logger.info("✓ Cross-check thành công! Kết quả dev khớp chính xác.")

    # 4. Xuất file vào --out-dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = out_dir / "bm25_seg_top100.jsonl"
    with open(candidates_path, "w", encoding="utf-8") as f:
        for r in dev_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in tune_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("✓ Đã ghi %d dòng xếp hạng vào %s", len(dev_records) + len(tune_records), candidates_path)

    # 5. Sao chép byte-identical bản dev và tune
    shutil.copyfile(dev_path, out_dir / "vlqa_dev500.jsonl")
    shutil.copyfile(tune_path, out_dir / "vlqa_tune1690.jsonl")
    logger.info("✓ Đã sao chép nguyên bản vlqa_dev500.jsonl và vlqa_tune1690.jsonl vào %s", out_dir)


if __name__ == "__main__":
    main()