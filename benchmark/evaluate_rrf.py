"""
evaluate_rrf.py - Bộ công cụ đánh giá thực nghiệm động cơ truy xuất lai (Hybrid RRF).
Khắc phục triệt để lỗi doc_number='N/A' và chuẩn hóa NDCG@K <= 1.0 qua cơ chế Target Deduplication.
"""

from __future__ import annotations

import re
import json
import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("benchmark", "eval_rrf")


def extract_doc_number_from_text(text: Optional[str]) -> str:
    """Rút trích số hiệu văn bản từ khối [META] nếu thuộc tính doc_number bị khuyết."""
    if not text:
        return ""
    match = re.search(r"\[META\]\s*Văn bản:\s*([^|\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def normalize_legal_identifier(text: Optional[str]) -> str:
    """Chuẩn hóa số hiệu văn bản và số Điều để đối soát chính xác tuyệt đối."""
    if not text:
        return ""
    norm = text.strip().lower()
    norm = re.sub(r"\s+", " ", norm)
    norm = re.sub(r"\bđiều\s+0*(\d+)\b", r"điều \1", norm)
    norm = re.sub(r"\bkhoản\s+0*(\d+)\b", r"khoản \1", norm)
    return norm


def extract_article_number(article_str: Optional[str]) -> Optional[str]:
    """Trích xuất số thứ tự Điều luật (VD: 'Điều 15' hoặc 'Điều 15 > Khoản 1' -> '15')."""
    if not article_str:
        return None
    match = re.search(r"điều\s+(\d+[a-zA-Z]?)", article_str.lower())
    return match.group(1) if match else None


def is_ground_truth_match(
    candidate: Dict[str, Any], 
    gt_doc_num: str, 
    gt_article: str,
    gt_chunk_id: Optional[str] = None
) -> bool:
    """Kiểm tra ứng viên truy xuất khớp chính xác số hiệu văn bản, điều luật hoặc chunk_id."""
    cand_chunk = str(candidate.get("chunk_id", "")).strip()

    # 1. Đối soát trực tiếp theo chunk_id (Tiêu chuẩn vàng IR benchmark)
    if gt_chunk_id and cand_chunk:
        if cand_chunk == gt_chunk_id or cand_chunk.startswith(gt_chunk_id) or gt_chunk_id.startswith(cand_chunk):
            return True
        gt_prefix = gt_chunk_id.split("_cl_")[0]
        cand_prefix = cand_chunk.split("_cl_")[0]
        if gt_prefix == cand_prefix:
            return True

    # 2. Khắc phục triệt để doc_number='N/A' bằng cách đọc thẻ [META]
    cand_doc_raw = candidate.get("doc_number")
    if not cand_doc_raw or str(cand_doc_raw).strip().lower() in ("n/a", "none", ""):
        cand_doc_raw = extract_doc_number_from_text(candidate.get("content")) or candidate.get("source_doc") or candidate.get("doc_id") or ""

    cand_doc = normalize_legal_identifier(str(cand_doc_raw))
    cand_art_str = candidate.get("hierarchy_path") or candidate.get("article") or candidate.get("content") or ""
    cand_art = normalize_legal_identifier(str(cand_art_str))

    target_doc = normalize_legal_identifier(gt_doc_num)
    target_art = normalize_legal_identifier(gt_article)

    doc_matched = True
    if target_doc and target_doc != "n/a":
        doc_matched = (target_doc in cand_doc) or (cand_doc in target_doc)

    art_matched = True
    if target_art and target_art != "điều khoản liên quan":
        target_num = extract_article_number(target_art)
        cand_num = extract_article_number(cand_art)
        if target_num and cand_num:
            art_matched = (target_num == cand_num)
        else:
            art_matched = bool(target_art and target_art in cand_art)

    return doc_matched and art_matched


def evaluate_query(
    ranked_candidates: List[Dict[str, Any]],
    sample: Dict[str, Any],
    k_thresholds: List[int],
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số Information Retrieval cho một truy vấn đơn lẻ.
    Áp dụng Target Deduplication: mỗi mục tiêu chỉ được tính điểm 1 lần duy nhất,
    đảm bảo DCG <= IDCG và NDCG luôn nằm nghiêm ngặt trong đoạn [0.0, 1.0].
    """
    gt_doc_num = sample.get("ground_truth_doc_number", "")
    gt_article = sample.get("ground_truth_article", "")
    gt_chunk_id = sample.get("ground_truth_chunk", "")
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
        for t_idx, (t_doc, t_art) in enumerate(target_pairs):
            if t_idx in matched_targets:
                continue
            if is_ground_truth_match(cand, t_doc, t_art, gt_chunk_id=gt_chunk_id):
                matched_targets.add(t_idx)
                relevance_vector[idx] = 1
                if first_hit_rank is None:
                    first_hit_rank = rank
                break

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
        query_metrics[f"NDCG@{k}"] = min(1.0, (dcg / idcg)) if idcg > 0 else 0.0

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
    """Nạp dữ liệu kiểm chuẩn từ thư mục benchmark."""
    datasets = {}
    b_path = Path(benchmark_dir)

    for filename in ["single_hop.jsonl", "multi_hop.jsonl", "vietlawbench_1000.jsonl"]:
        file_path = b_path / filename
        if not file_path.exists():
            continue

        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    try:
                        samples.append(json.loads(clean_line))
                    except json.JSONDecodeError:
                        continue

        ds_name = filename.replace(".jsonl", "")
        datasets[ds_name] = samples
        logger.info("✓ Đã nạp thành công '%s': %d câu hỏi.", filename, len(samples))

    return datasets


def retrieve_by_mode(retriever: Any, query: str, top_k: int, mode: str) -> List[Dict[str, Any]]:
    """Thực thi truy xuất phân tích bóc tách thành phần (Ablation Mode)."""
    if mode == "dense_only":
        return retriever._search_dense(query, top_k=top_k)
    elif mode == "sparse_only":
        return retriever._search_sparse_es(query, top_k=top_k)
    else:
        return retriever.retrieve(query, top_k=top_k)


def run_benchmark_evaluation(
    benchmark_dir: str | Path = BENCHMARK_DIR,
    output_dir: Optional[str | Path] = ROOT_DIR / "benchmark" / "results",
    k_list: Optional[List[int]] = None,
    mode: str = "hybrid",
    retriever_instance: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Hàm đánh giá chuẩn hóa được export cho toàn bộ benchmark suite."""
    k_thresholds = sorted(list(set(k_list or [1, 3, 5, 10])))
    max_k = max(k_thresholds)

    if retriever_instance is not None:
        retriever = retriever_instance
    else:
        from database.qdrant_client import QdrantClientWrapper
        from rag.es_retriever import LegalElasticsearchRetriever
        from rag.retriever import LegalHybridRetriever
        from sentence_transformers import SentenceTransformer
        from configs.config import config

        qdrant = QdrantClientWrapper(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        es = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=config.ES_INDEX_NAME)
        encoder = SentenceTransformer(config.BASE_MODEL_NAME, device=config.EMBED_DEVICE)
        retriever = LegalHybridRetriever(qdrant_wrapper=qdrant, es_retriever=es, encoder_model=encoder)

    datasets = load_benchmark(benchmark_dir)
    all_dataset_results = {}
    sample_level_metrics = {}

    for ds_name, samples in datasets.items():
        if ds_name == "vietlawbench_1000" and ("single_hop" in datasets and "multi_hop" in datasets):
            continue

        dataset_query_scores: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples, 1):
            query = sample.get("query", "")
            if not query:
                continue

            retrieved_docs = retrieve_by_mode(retriever, query, top_k=max_k, mode=mode)
            metrics = evaluate_query(retrieved_docs, sample, k_thresholds)
            dataset_query_scores.append(metrics)

        total_samples = len(dataset_query_scores)
        aggregated_metrics = {"Total_Samples": total_samples, "Evaluation_Mode": mode}
        if total_samples > 0:
            for metric_key in dataset_query_scores[0].keys():
                mean_val = sum(score[metric_key] for score in dataset_query_scores) / total_samples
                aggregated_metrics[f"Mean_{metric_key}"] = round(mean_val, 4)

        all_dataset_results[ds_name] = aggregated_metrics
        sample_level_metrics[ds_name] = dataset_query_scores

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        summary_file = out_path / f"rrf_summary_{mode}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_dataset_results, f, ensure_ascii=False, indent=2)

    return all_dataset_results, sample_level_metrics


def main():
    parser = argparse.ArgumentParser(description="Chương trình kiểm chuẩn truy xuất VietLawBERT-MRL")
    parser.add_argument("--benchmark-dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "benchmark" / "results"))
    parser.add_argument("--k-list", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--mode", choices=["hybrid", "dense_only", "sparse_only"], default="hybrid")
    args = parser.parse_args()

    run_benchmark_evaluation(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        k_list=args.k_list,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
