"""
demo/server/generation_service.py — Microservice: Generation
=============================================================
Singleton wrapper cho LegalGenerator.
Nhận pre-retrieved docs từ retrieval_service để tránh truy xuất 2 lần
(LegalGenerator.ask() hỗ trợ tham số contexts= cho phép điều này).

Khi use_mock=True: trả về câu trả lời giả để dev/test UI.
"""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("demo.generation_service")

# ─── Singleton state ────────────────────────────────────────────────────────
_generator = None


def _get_generator():
    """Khởi tạo LegalGenerator một lần duy nhất (Singleton)."""
    global _generator
    if _generator is None:
        logger.info("Đang khởi tạo LegalGenerator (ResilientLLMDispatcher)...")
        from rag.generator import LegalGenerator
        _generator = LegalGenerator()
        logger.info("LegalGenerator sẵn sàng.")
    return _generator


# ─── Mock generation ────────────────────────────────────────────────────────

def _mock_generate(query: str, docs: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """Mock generation — trả về câu trả lời tĩnh cho dev UI."""
    time.sleep(0.2)
    if not docs:
        answer = "Cơ sở dữ liệu hiện tại không đủ căn cứ pháp lý để giải đáp thắc mắc này."
        return {
            "answer": answer,
            "attribution_score": 0.0,
            "model_used": "Mock",
            "latency_ms": 200.0,
            "mock": True,
        }

    d = docs[0]
    answer = (
        f"Căn cứ theo {d.get('doc_number', 'văn bản liên quan')}, "
        f"tại {d.get('hierarchy_path', 'điều khoản liên quan')} quy định:\n\n"
        f"{d.get('content', d.get('text', ''))}\n\n"
        f"*(Chế độ: {mode} | [MOCK — chưa kết nối LLM thật])*"
    )
    return {
        "answer": answer,
        "attribution_score": 0.67,
        "model_used": "Mock",
        "latency_ms": 201.0,
        "mock": True,
    }


# ─── Public API ─────────────────────────────────────────────────────────────

def generate(
    query: str,
    docs: List[Dict[str, Any]],
    mode: str = "Full (+ Graph Rerank)",
    use_mock: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Sinh câu trả lời pháp lý từ docs đã truy xuất.

    Parameters
    ----------
    query      : câu hỏi gốc của người dùng
    docs       : danh sách chunk từ retrieval_service.retrieve()["docs"]
    mode       : chế độ ablation (chỉ dùng cho hiển thị)
    use_mock   : True → trả về câu trả lời giả
    temperature: độ sáng tạo LLM
    max_tokens : giới hạn token đầu ra

    Returns
    -------
    {
        "answer"            : str,
        "attribution_score" : float,  # RQ4: tỷ lệ căn cứ được trích dẫn
        "model_used"        : str,    # tên LLM provider thực tế
        "latency_ms"        : float,
        "mock"              : bool,
    }
    """
    if use_mock:
        return _mock_generate(query, docs, mode)

    t0 = time.perf_counter()
    generator = _get_generator()

    try:
        # Truyền contexts= để tái sử dụng docs đã retrieved → tránh double-fetch
        result = generator.ask(
            query=query,
            top_k=len(docs) or 3,
            contexts=docs if docs else None,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "answer"            : result.get("answer", ""),
            "attribution_score" : result.get("attribution_score", 0.0),
            "model_used"        : result.get("model_used", "Unknown"),
            "latency_ms"        : latency_ms,
            "mock"              : False,
        }

    except Exception as exc:
        logger.error("Generation thất bại: %s", exc)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "answer"            : f"Lỗi hệ thống khi tổng hợp câu trả lời: {exc}",
            "attribution_score" : 0.0,
            "model_used"        : "Error",
            "latency_ms"        : latency_ms,
            "mock"              : False,
            "error"             : str(exc),
        }
