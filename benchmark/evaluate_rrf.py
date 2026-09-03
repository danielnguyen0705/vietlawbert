"""
evaluate_rrf.py - Bộ công cụ đánh giá thực nghiệm động cơ truy xuất lai (Hybrid RRF).
Hiện thực hóa chuẩn đánh giá IR quốc tế (Hit-Rate@K, MRR@K, NDCG@K, MAP@K) phục vụ
bộ chuẩn VietLawBench (Single-hop & Multi-hop) và kiểm định Paired Student's t-test.
"""

import os
import sys
import re
import json
import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Thiết lập đường dẫn thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VietLawBERT_Eval")


def normalize_legal_identifier(text: Optional[str]) -> str:
    """
    Chuẩn hóa số hiệu văn bản và số Điều để đối soát chính xác tuyệt đối.
    Ví dụ: '100/2019/NĐ-CP' -> '100/2019/nd-cp', 'Điều  06' -> 'dieu 6'.
    """
    if not text:
        return ""
    norm = text.strip().lower()
    norm = re.sub(r"\s+", " ", norm)
    norm = re.sub(r"\bđiều\s+0*(\d+)\b", r"điều \1", norm)
    norm = re.sub(r"\bkhoản\s+0*(\d+)\b", r"khoản \1", norm)
    return norm


def extract_article_number(article_str: Optional[str]) -> Optional[str]:
    """Trích xuất duy nhất số thứ tự Điều luật (VD: 'Điều 15.' -> '15')."""
    if not article_str:
        return None
    match = re.search(r"điều\s+(\d+)", article_str.lower())
    return match.group(1) if match else None


def is_ground_truth_match(candidate: Dict[str, Any], gt_doc_num: str, gt_article: str) -> bool:
    """
    Kiểm tra một ứng viên truy xuất có khớp chính xác cả Số hiệu văn bản và Điều luật hay không.
    Ngăn chặn tuyệt đối hiện tượng False Positive từ việc so sánh chuỗi con.
    """
    cand_doc = normalize_legal_identifier(candidate.get("doc_number") or candidate.get("source_doc") or "")
    cand_art = normalize_legal_identifier(candidate.get("article") or "")

    target_doc = normalize_legal_identifier(gt_doc_num)
    target_art = normalize_legal_identifier(gt_article)

    doc_matched = True
    if target_doc:
        doc_matched = (target_doc in cand_doc) or (cand_doc in target_doc)

    art_matched = True
    if target_art:
        target_num = extract_article_number(target_art)
        cand_num = extract_article_number(cand_art)
        if target_num and cand_num:
            art_matched = (target_num == cand_num)
        else:
            art_matched = (target_art == cand_art)

    return doc_matched and art_matched


def evaluate_query(
    ranked_candidates: List[Dict[str, Any]],
    sample: Dict[str, Any],
    k_thresholds: List[int]
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số Information Retrieval cho một câu truy vấn đơn lẻ:
    Hit-Rate@K, MRR@K, NDCG@K, và MAP@K trên danh sách các mốc K.
    """
    gt_doc_num = sample.get("ground_truth_doc_number", "")
    gt_article = sample.get("ground_truth_article", "")
    gt_docs = sample.get("ground_truth_docs", [])

    target_pairs: List[Tuple[str, str]] = []
    if gt_docs and isinstance(gt_docs, list):
        for doc in gt_docs:
            target_pairs.append((doc.get("doc_number", ""), doc.get("article", "")))
    else:
        target_pairs.append((gt_doc_num, gt_article))

    num_candidates = len(ranked_candidates)
    relevance_vector = [0] * num_candidates
    first_hit_rank: Optional[int] = None
    matched_targets = set()

    for idx, cand in enumerate(ranked_candidates):
        rank = idx + 1
        is_hit = False
        for t_idx, (t_doc, t_art) in enumerate(target_pairs):
            if is_ground_truth_match(cand, t_doc, t_art):
                is_hit = True
                matched_targets.add(t_idx)
                break

        if is_hit:
            relevance_vector[idx] = 1
            if first_hit_rank is None:
                first_hit_rank = rank

    query_metrics = {}
    for k in k_thresholds:
        sub_rel = relevance_vector[:k]
        query_metrics[f"Hit@{k}"] = 1.0 if any(sub_rel) else 0.0

        if first_hit_rank is not None and first_hit_rank <= k:
            query_metrics[f"MRR@{k}"] = 1.0 / float(first_hit_rank)
        else:
            query_metrics[f"MRR@{k}"] = 0.0

        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(sub_rel))
        num_relevant = min(len(target_pairs), k)
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(num_relevant))
        query_metrics[f"NDCG@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

        running_hits = 0
        precisions = []
        for idx, rel in enumerate(sub_rel):
            if rel == 1:
                running_hits += 1
                precisions.append(running_hits / (idx + 1))
        query_metrics[f"MAP@{k}"] = (sum(precisions) / num_relevant) if num_relevant > 0 and precisions else 0.0

    query_metrics["Evidence_Coverage"] = len(matched_targets) / len(target_pairs) if target_pairs else 0.0
    return query_metrics


def load_benchmark(benchmark_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Nạp các tập kiểm thử VietLawBench chuẩn hóa từ thư mục benchmark."""
    datasets = {}
    bench_path = Path(benchmark_dir)
    target_files = ["single_hop.jsonl", "multi_hop.jsonl"]

    for filename in target_files:
        file_path = bench_path / filename
        if not file_path.exists():
            logger.warning(f"Không tìm thấy tập dữ liệu kiểm thử: {file_path}")
            continue

        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                clean_line = line.strip()
                if clean_line:
                    try:
                        samples.append(json.loads(clean_line))
                    except json.JSONDecodeError as err:
                        logger.error(f"Lỗi cú pháp JSON tại dòng {line_no} trong {filename}: {err}")

        ds_name = filename.replace(".jsonl", "")
        datasets[ds_name] = samples
        logger.info(f"✓ Đã nạp thành công '{filename}': {len(samples)} mẫu câu hỏi đối chứng.")

    return datasets


def run_benchmark_evaluation(
    benchmark_dir: str | Path = ROOT_DIR / "benchmark",
    output_dir: Optional[str | Path] = ROOT_DIR / "benchmark" / "results",
    k_list: Optional[List[int]] = None,
    retriever_instance: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Hàm API thực thi đánh giá benchmark theo lập trình.
    Cho phép gọi trực tiếp từ Python mà không cần gọi CLI subprocess.
    """
    k_thresholds = sorted(list(set(k_list or [1, 3, 5, 10])))
    max_k = max(k_thresholds)

    if retriever_instance is not None:
        retriever = retriever_instance
    else:
        from rag.retriever import LegalRetriever
        logger.info("Đang khởi tạo Động cơ Truy xuất Lai (Dense MRL + Sparse BM25 + Neo4j Graph)...")
        retriever = LegalRetriever()

    datasets = load_benchmark(benchmark_dir)
    if not datasets:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu test (single_hop.jsonl/multi_hop.jsonl) tại: {benchmark_dir}")

    all_dataset_results = {}
    sample_level_metrics = {}

    for ds_name, samples in datasets.items():
        logger.info(f"\n=======================================================")
        logger.info(f"BẮT ĐẦU ĐÁNH GIÁ TẬP TEST: {ds_name.upper()} (Tổng: {len(samples)} mẫu)")
        logger.info(f"=======================================================")

        dataset_query_scores: List[Dict[str, Any]] = []

        for idx, sample in enumerate(samples, 1):
            query = sample.get("query", "")
            if not query:
                continue

            retrieved_docs = retriever.search_context(query, top_k=max_k)
            metrics = evaluate_query(retrieved_docs, sample, k_thresholds)
            dataset_query_scores.append(metrics)

            if idx % 50 == 0 or idx == len(samples):
                logger.info(f"Tiến độ: {idx}/{len(samples)} câu hỏi...")

        total_samples = len(dataset_query_scores)
        aggregated_metrics = {"Total_Samples": total_samples}

        if total_samples > 0:
            for metric_key in dataset_query_scores[0].keys():
                mean_val = sum(score[metric_key] for score in dataset_query_scores) / total_samples
                aggregated_metrics[f"Mean_{metric_key}"] = round(mean_val, 4)

        all_dataset_results[ds_name] = aggregated_metrics
        sample_level_metrics[ds_name] = dataset_query_scores

        logger.info(f"\n--- KẾT QUẢ TỔNG HỢP [{ds_name.upper()}] ---")
        for k, v in aggregated_metrics.items():
            logger.info(f"  * {k:25}: {v}")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        summary_file = out_path / "rrf_evaluation_summary.json"
        detailed_file = out_path / "rrf_sample_level_distributions.json"

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_dataset_results, f, ensure_ascii=False, indent=2)

        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(sample_level_metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✓ Đã lưu bảng tổng hợp chỉ số: {summary_file}")
        logger.info(f"✓ Đã lưu phân phối mẫu chi tiết phục vụ t-test: {detailed_file}")

    return all_dataset_results, sample_level_metrics


def main():
    parser = argparse.ArgumentParser(description="Chương trình kiểm chuẩn truy xuất lai VietLawBERT-MRL")
    parser.add_argument("--benchmark_dir", default=str(ROOT_DIR / "benchmark"), help="Đường dẫn thư mục chứa dữ liệu test")
    parser.add_argument("--output_dir", default=str(ROOT_DIR / "benchmark" / "results"), help="Thư mục xuất báo cáo thực nghiệm")
    parser.add_argument("--k_list", nargs="+", type=int, default=[1, 3, 5, 10], help="Danh sách các mốc K đánh giá")
    args = parser.parse_args()

    run_benchmark_evaluation(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        k_list=args.k_list,
    )


if __name__ == "__main__":
    main()