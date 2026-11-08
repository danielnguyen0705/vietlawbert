"""
benchmark/metrics_article.py — Các hàm đo chất lượng truy xuất cấp điều luật (Article-level IR metrics).

Toần hàm thuần (pure Python), không phụ thuộc Elasticsearch hay numpy.
Đầu vào:
  ranked: list[str] — danh sách chunk_id theo thứ tự xếp hạng (đã lọc trùng)
  gold: set[str]    — tập chunk_id nhãn đúng (ground-truth)
"""
from __future__ import annotations

import math
from typing import List, Set, Dict, Any


def hit_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """1.0 nếu có ít nhất một chunk trong top-k thuộc gold, ngược lại 0.0."""
    if not gold or k <= 0:
        return 0.0
    top_k = ranked[:k]
    return 1.0 if any(item in gold for item in top_k) else 0.0


def recall_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """|top-k ∩ gold| / |gold|"""
    if not gold or k <= 0:
        return 0.0
    top_k_set = set(ranked[:k])
    return len(top_k_set & gold) / len(gold)


def precision_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """|top-k ∩ gold| / k"""
    if not gold or k <= 0:
        return 0.0
    top_k_set = set(ranked[:k])
    return len(top_k_set & gold) / k


def mrr_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """1 / rank của phần tử relevant đầu tiên trong top-k (rank bắt đầu từ 1), else 0.0."""
    if not gold or k <= 0:
        return 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in gold:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """
    NDCG nhị phân:
      DCG = Σ (rel_i / log2(i + 1)) với i từ 1 đến k (rel_i ∈ {0, 1})
      IDCG = Σ (1 / log2(i + 1)) với i từ 1 đến min(|gold|, k)
    """
    if not gold or k <= 0:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in gold:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_count = min(len(gold), k)
    if ideal_count == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """
    Average Precision @ k:
      AP@k = (Σ (precision@i * rel_i) với i từ 1 đến k) / min(|gold|, k)
    """
    if not gold or k <= 0:
        return 0.0
    hits = 0
    prec_sum = 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in gold:
            hits += 1
            prec_sum += hits / idx
    ideal = min(len(gold), k)
    return prec_sum / ideal if ideal > 0 else 0.0


def f2_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """
    F2 score: nhấn mạnh Recall (beta = 2).
      F2 = 5 * P * R / (4 * P + R)
    Trả về 0.0 nếu P + R == 0.
    """
    if not gold or k <= 0:
        return 0.0
    p = precision_at_k(ranked, gold, k)
    r = recall_at_k(ranked, gold, k)
    denom = 4.0 * p + r
    if denom == 0.0:
        return 0.0
    return (5.0 * p * r) / denom


def aggregate(per_query: List[Dict[str, Any]]) -> Dict[str, float]:
    """Tính trung bình cộng (macro average) cho mọi key số thực/nguyên trong danh sách per_query."""
    if not per_query:
        return {}
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in per_query:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    return {k: round(totals[k] / counts[k], 6) for k in totals}
