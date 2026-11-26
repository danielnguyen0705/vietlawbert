"""
eval_utils.py - Tiện ích thuần (numpy-only, không kéo torch/sbert/neo4j) dùng chung cho
huấn luyện và đánh giá. Đặt ngoài package benchmark để training không phụ thuộc sentence_transformers.

- minmax_norm      : chuẩn hóa Min-Max theo từng truy vấn cho điểm BM25 (findings B2).
- mrr_at_k         : MRR@k trên ma trận tương đồng (chọn best checkpoint theo MRR, findings A2).
- build_candidate_pool : kho ứng viên có hard-negative BM25 cùng chủ đề (chống bão hòa 1.0, findings C1).
- ALL_DIMS         : dải lát cắt Matryoshka chuẩn.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("eval_utils")

ALL_DIMS = [64, 128, 256, 512, 768, 1024]


def minmax_norm(scores: List[float]) -> List[float]:
    """Chuẩn hóa Min-Max về [0,1] theo từng tập (findings B2). Tập rỗng -> rỗng; mọi giá trị bằng nhau -> 0."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    return [(float(x) - lo) / span for x in scores]


def mrr_at_k(sim: np.ndarray, k: int = 10) -> float:
    """MRR@k cho ma trận vuông sim (n x n): dòng i là độ tương đồng của anchor i với mọi positive,
    gold của anchor i nằm ở cột i (đường chéo). Dùng chọn checkpoint tốt nhất trên tập validation."""
    n = sim.shape[0]
    if n == 0:
        return 0.0
    order = np.argsort(-sim, axis=1)
    total = 0.0
    for i in range(n):
        found = np.where(order[i] == i)[0]
        rank = int(found[0]) + 1 if found.size else n + 1
        if rank <= k:
            total += 1.0 / rank
    return total / n


def build_candidate_pool(
    samples: List[Dict[str, Any]],
    es: Optional[Any] = None,
    per_query_hard: int = 20,
) -> List[Dict[str, str]]:
    """Kho ứng viên = gold của mỗi câu + hard-negative BM25 cùng chủ đề lấy từ ES (findings C1).
    Không có ES hoặc ES trả rỗng -> chỉ còn gold, CẢNH BÁO vì điểm dễ bão hòa 1.0."""
    pool: List[Dict[str, str]] = []
    seen: set = set()

    def _add(cid: Any, content: Any, doc: str = "", art: str = "") -> None:
        cid = str(cid or "").strip()
        content = str(content or "").strip()
        if cid and content and cid not in seen:
            seen.add(cid)
            pool.append({"chunk_id": cid, "doc_number": str(doc or ""), "article": str(art or ""), "content": content})

    for s in samples:
        _add(
            s.get("ground_truth_chunk") or f"gold_{len(seen)}",
            s.get("raw_content") or s.get("evidence_text", "N/A"),
            s.get("ground_truth_doc_number", ""),
            s.get("ground_truth_article", ""),
        )

    added = 0
    if es is not None:
        for s in samples:
            try:
                for h in es.search(s.get("query", ""), top_k=per_query_hard):
                    before = len(seen)
                    _add(h.get("chunk_id") or h.get("id"), h.get("content", ""),
                         h.get("doc_number", ""), h.get("article", ""))
                    added += len(seen) - before
            except Exception as ex:
                logger.warning("Không thêm được hard-negative từ ES (%s).", ex)
                break

    if added == 0:
        logger.warning("Pool chỉ gồm %d gold, KHÔNG có hard-negative -> NGUY CƠ bão hòa điểm 1.0.", len(pool))
    else:
        logger.info("Pool ứng viên: %d chunk (%d hard-negative BM25 cùng chủ đề).", len(pool), added)
    return pool
