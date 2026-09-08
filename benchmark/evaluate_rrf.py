"""
evaluate_rrf.py - Bộ công cụ đánh giá thực nghiệm động cơ truy xuất lai (Hybrid RRF).
Hiện thực hóa chuẩn đánh giá IR quốc tế (Hit-Rate@K, MRR@K, NDCG@K, MAP@K) phục vụ
bộ chuẩn VietLawBench (Single-hop & Multi-hop), hỗ trợ Ablation Study.
"""

from __future__ import annotations

import os
import sys
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
    """Trích xuất duy nhất số thứ tự Điều luật (VD: 'Điều 15.' hoặc 'Điều 15 > Khoản 1' -> '15')."""
    if not article_str:
        return None
    match = re.search(r"điều\s+(\d+[a-zA-Z]?)", article_str.lower())
    return match.group(1) if match else None


def is_ground_truth_match(candidate: Dict[str, Any], gt_doc_num: str, gt_article: str) -> bool:
    """
    Kiểm tra ứng viên truy xuất có khớp chính xác cả Số hiệu văn bản và Điều luật.
    Bóc tách linh hoạt từ doc_number, hierarchy_path, content.
    """
    cand_doc = normalize_legal_identifier(
        candidate.get("doc_number") or candidate.get("source_doc") or candidate.get("doc_id") or ""
    )
    # Bóc tách Điều luật từ hierarchy_path (ví dụ: 'Điều 6 > Khoản 3') hoặc trường article
    cand_art_str = candidate.get("hierarchy_path") or candidate.get("article") or ""
    cand_art = normalize_legal_identifier(cand_art_str)

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
            art_matched = (target_art in cand_art)

    return doc_matched and art_matched


def evaluate_query(
    ranked_candidates: List[Dict[str, Any]],
    sample: Dict[str, Any],
    k_thresholds: List[int],
) -> Dict[str, Any]:
    """Tính toán các chỉ số Information Retrieval cho một truy vấn đơn lẻ."""
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
        logger.info(f"✓ Đã nạp thành công '{filename}': {len(samples)} câu hỏi đối chứng.")

    return datasets


def retrieve_by_mode(retriever: Any, query: str, top_k: int, mode: str) -> List[Dict[str, Any]]:
    """Thực thi truy xuất phân tích bóc tách thành phần (Ablation Mode) tương thích v3."""
    if mode == "dense_only":
        return retriever._search_dense(query, top_k=top_k)
    elif mode == "sparse_only":
        return retriever._search_sparse_es(query, top_k=top_k)
    else:
        # Mặc định: Full Hybrid RRF + Graph Reranking
        return retriever.retrieve(query, top_k=top_k)


def run_benchmark_evaluation(
    benchmark_dir: str | Path = BENCHMARK_DIR,
    output_dir: Optional[str | Path] = ROOT_DIR / "benchmark" / "results",
    k_list: Optional[List[int]] = None,
    mode: str = "hybrid",
    retriever_instance: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    k_thresholds = sorted(list(set(k_list or [1, 3, 5, 10])))
    max_k = max(k_thresholds)

    if retriever_instance is not None:
        retriever = retriever_instance
    else:
        # Khởi tạo retriever lai kết nối Qdrant và Elasticsearch
        from database.qdrant_client import QdrantClientWrapper
        from rag.es_retriever import LegalElasticsearchRetriever
        from rag.retriever import LegalHybridRetriever
        from sentence_transformers import SentenceTransformer
        from configs.config import config

        qdrant = QdrantClientWrapper(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        es = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=config.ES_INDEX_NAME)
        encoder = SentenceTransformer(config.BASE_MODEL_NAME)
        retriever = LegalHybridRetriever(qdrant_wrapper=qdrant, es_client=es.client, encoder_model=encoder)

    datasets = load_benchmark(benchmark_dir)
    if not datasets:
        logger.warning("Chưa có tập benchmark, đang kích hoạt sinh mẫu tự động...")
        from benchmark.build_vietlawbench import VietLawBenchBuilder
        builder = VietLawBenchBuilder()
        builder.build_and_export_all(600, 400)
        builder.close()
        datasets = load_benchmark(benchmark_dir)

    all_dataset_results = {}
    sample_level_metrics = {}

    for ds_name, samples in datasets.items():
        if ds_name == "vietlawbench_1000" and ("single_hop" in datasets and "multi_hop" in datasets):
            continue

        logger.info(f"\n=======================================================")
        logger.info(f"ĐÁNH GIÁ: {ds_name.upper()} | Chế độ: [{mode.upper()}] | Số mẫu: {len(samples)}")
        logger.info(f"=======================================================")

        dataset_query_scores: List[Dict[str, Any]] = []

        for idx, sample in enumerate(samples, 1):
            query = sample.get("query", "")
            if not query:
                continue

            retrieved_docs = retrieve_by_mode(retriever, query, top_k=max_k, mode=mode)
            metrics = evaluate_query(retrieved_docs, sample, k_thresholds)
            dataset_query_scores.append(metrics)

            if idx % 50 == 0 or idx == len(samples):
                logger.info(f"Tiến độ: {idx}/{len(samples)} câu hỏi...")

        total_samples = len(dataset_query_scores)
        aggregated_metrics = {"Total_Samples": total_samples, "Evaluation_Mode": mode}

        if total_samples > 0:
            for metric_key in dataset_query_scores[0].keys():
                mean_val = sum(score[metric_key] for score in dataset_query_scores) / total_samples
                aggregated_metrics[f"Mean_{metric_key}"] = round(mean_val, 4)

        all_dataset_results[ds_name] = aggregated_metrics
        sample_level_metrics[ds_name] = dataset_query_scores

        logger.info(f"\n--- KẾT QUẢ ĐÁNH GIÁ [{ds_name.upper()} - {mode.upper()}] ---")
        for k, v in aggregated_metrics.items():
            logger.info(f"  * {k:25}: {v}")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        summary_file = out_path / f"rrf_summary_{mode}.json"
        detailed_file = out_path / f"rrf_distribution_{mode}.json"

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_dataset_results, f, ensure_ascii=False, indent=2)

        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(sample_level_metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✓ Đã lưu bảng chỉ số thực nghiệm: {summary_file}")

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