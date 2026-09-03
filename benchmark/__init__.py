"""
vietlawbert.benchmark
~~~~~~~~~~~~~~~~~~~~~
Bộ công cụ kiểm chuẩn và đánh giá thực nghiệm chuẩn học thuật Q1 (VietLawBench).
Đo đạc các chỉ số IR chuẩn hóa: Hit-Rate@K, MRR@K, NDCG@K, MAP@K trên tập Single-hop và Multi-hop.
"""

from .evaluate_rrf import (
    load_benchmark,
    evaluate_query,
    is_ground_truth_match,
    normalize_legal_identifier,
    extract_article_number,
    run_benchmark_evaluation,
)

__all__ = [
    "load_benchmark",
    "evaluate_query",
    "is_ground_truth_match",
    "normalize_legal_identifier",
    "extract_article_number",
    "run_benchmark_evaluation",
]