"""
demo/server/retrieval_service.py — Microservice: Retrieval
===========================================================
Singleton wrapper cho LegalHybridRetriever.
Hỗ trợ ablation qua tham số mode:
  - "Dense"               → chỉ gọi _search_dense()
  - "Sparse (BM25)"       → chỉ gọi _search_sparse_es()
  - "Hybrid-RRF"          → RRF fusion, bỏ qua graph (alpha=0)
  - "Full (+ Graph Rerank)" → pipeline đầy đủ (alpha=config.GRAPH_ALPHA)

NOTE: Khi USE_MOCK_BACKEND=true, trả về dữ liệu giả để dev/test UI
mà không cần Qdrant/ES running. Đặt env var USE_MOCK_BACKEND=false
khi services thật đã sẵn sàng.
"""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("demo.retrieval_service")

# ─── Singleton state ────────────────────────────────────────────────────────
_retriever = None


def _get_retriever():
    """Khởi tạo LegalHybridRetriever một lần duy nhất (Singleton)."""
    global _retriever
    if _retriever is None:
        logger.info("Đang khởi tạo LegalHybridRetriever...")
        from rag.retriever import LegalHybridRetriever
        _retriever = LegalHybridRetriever()
        logger.info("LegalHybridRetriever sẵn sàng.")
    return _retriever


# ─── Mock data (WP11.1 — UI dev mode) ──────────────────────────────────────

MOCK_DOCS: List[Dict[str, Any]] = [
    {
        "chunk_id": "mock-001",
        "doc_number": "Luật Doanh nghiệp 59/2020/QH14",
        "hierarchy_path": "Luật DN 2020 > Chương III > Điều 17 > Khoản 2",
        "content": (
            "Doanh nghiệp có quyền kinh doanh các ngành, nghề mà pháp luật "
            "không cấm và chịu trách nhiệm về hoạt động của mình theo quy định "
            "của pháp luật."
        ),
        "effective_date": "01/01/2021",
        "final_rerank_score": 0.92,
        "graph_boost": 0.08,
        "rrf_score": 0.0167,
    },
    {
        "chunk_id": "mock-002",
        "doc_number": "Nghị định 01/2021/NĐ-CP",
        "hierarchy_path": "NĐ 01/2021 > Chương IV > Điều 23 > Khoản 1",
        "content": (
            "Hồ sơ đăng ký doanh nghiệp bao gồm: Giấy đề nghị đăng ký doanh "
            "nghiệp; Điều lệ công ty (đối với công ty); Danh sách thành viên "
            "hoặc danh sách cổ đông sáng lập."
        ),
        "effective_date": "04/01/2021",
        "final_rerank_score": 0.85,
        "graph_boost": 0.05,
        "rrf_score": 0.0154,
    },
    {
        "chunk_id": "mock-003",
        "doc_number": "Thông tư 01/2021/TT-BKHĐT",
        "hierarchy_path": "TT 01/2021 > Chương II > Điều 8",
        "content": (
            "Người thành lập doanh nghiệp hoặc người đại diện theo pháp luật "
            "nộp hồ sơ đăng ký doanh nghiệp tại Cơ quan đăng ký kinh doanh "
            "nơi doanh nghiệp đặt trụ sở chính."
        ),
        "effective_date": "15/01/2021",
        "final_rerank_score": 0.78,
        "graph_boost": 0.03,
        "rrf_score": 0.0143,
    },
]


def _mock_retrieve(query: str, top_k: int, mode: str) -> Dict[str, Any]:
    """Mock retrieval — trả về docs giả, kèm latency giả lập."""
    time.sleep(0.15)
    docs = MOCK_DOCS[:top_k]
    return {
        "docs": docs,
        "latency_ms": 142.5,
        "mode": mode,
        "top_k": top_k,
        "mock": True,
    }


# ─── Public API ─────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = 5,
    mode: str = "Full (+ Graph Rerank)",
    use_mock: bool = False,
) -> Dict[str, Any]:
    """
    Thực hiện truy xuất pháp lý theo chế độ ablation.

    Returns
    -------
    {
        "docs"       : List[Dict],   # danh sách chunk truy xuất được
        "latency_ms" : float,        # thời gian truy xuất (ms)
        "mode"       : str,          # chế độ đã dùng
        "top_k"      : int,
        "mock"       : bool,         # True nếu dùng mock backend
    }
    """
    if use_mock:
        return _mock_retrieve(query, top_k, mode)

    t0 = time.perf_counter()
    retriever = _get_retriever()

    try:
        if mode == "Dense":
            raw_docs = retriever._search_dense(query, top_k=top_k, must_be_effective=True)
            docs = raw_docs[:top_k]

        elif mode == "Sparse (BM25)":
            raw_docs = retriever._search_sparse_es(query, top_k=top_k, must_be_effective=True)
            docs = raw_docs[:top_k]

        elif mode == "Hybrid-RRF":
            original_alpha = retriever.alpha
            retriever.alpha = 0.0
            docs = retriever.retrieve(query, top_k=top_k, must_be_effective=True)
            retriever.alpha = original_alpha

        else:
            # "Full (+ Graph Rerank)" — pipeline đầy đủ
            docs = retriever.retrieve(query, top_k=top_k, must_be_effective=True)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "docs": docs,
            "latency_ms": latency_ms,
            "mode": mode,
            "top_k": top_k,
            "mock": False,
        }

    except Exception as exc:
        logger.error("Retrieval thất bại (mode=%s): %s", mode, exc)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "docs": [],
            "latency_ms": latency_ms,
            "mode": mode,
            "top_k": top_k,
            "mock": False,
            "error": str(exc),
        }
