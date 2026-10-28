"""
demo/server/interaction_service.py — Microservice: Interaction / Session Logging
================================================================================
Ghi lại TOÀN BỘ mỗi lượt hỏi-đáp (không chỉ feedback like/dislike) vào JSONL,
phục vụ:
  - Phân tích định lượng chất lượng truy xuất & sinh (RQ1–RQ4)
  - Tái lập bằng chứng (reproducible evidence) cho báo cáo / paper
  - Xây dựng bộ test hồi quy (regression test) cho pipeline RAG

File: flagging/interactions.jsonl  (gitignored — dữ liệu runtime, KHÔNG commit)
Mỗi dòng = 1 lượt truy vấn. Schema:
{
  "timestamp"  : "2026-10-28T16:00:00.000+00:00",
  "run_id"     : "a1b2c3d4",          # id của 1 lần chạy demo (1 process)
  "turn_index" : 0,                   # thứ tự lượt trong phiên
  "query"      : "...",
  "answer"     : "...",
  "mode"       : "Full (+ Graph Rerank)",
  "top_k"      : 5,
  "model_used" : "Primary_Cloud_SOTA",
  "is_mock"    : false,
  "metrics"    : {
      "retrieval_latency_ms"  : 123.4,
      "generation_latency_ms" : 2345.6,
      "total_latency_ms"      : 2469.0,
      "attribution_score"     : 0.33,
      "n_docs"                : 5
  },
  "evidence"   : [                    # top-k bằng chứng, đầy đủ nội dung + điểm số
      {
        "rank"           : 1,
        "doc_number"     : "134/2026/NĐ-CP",
        "hierarchy_path" : "Điều 4 (...) > Khoản 1",
        "effective_date" : "2026-04-09T00:00:00",
        "content"        : "...toàn văn nội dung chunk...",
        "rerank_score"   : 0.016,
        "graph_boost"    : 0.0,
        "rrf_score"      : 0.0164
      },
      ...
  ]
}
"""

from __future__ import annotations

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from demo.config import FLAGGING_DIR, INTERACTIONS_FILE

logger = logging.getLogger("demo.interaction_service")

# ID cho mỗi lần chạy demo (1 process). Giúp nhóm các lượt cùng một phiên khi phân tích.
RUN_ID = uuid.uuid4().hex[:8]


def _ensure_flagging_dir() -> None:
    FLAGGING_DIR.mkdir(parents=True, exist_ok=True)


def _extract_evidence(docs: List[Any]) -> List[Dict[str, Any]]:
    """
    Trích xuất các trường bằng chứng + điểm số từ mỗi doc.
    Dùng ĐÚNG các khóa mà panels.render_evidence đọc → log khớp 100% với UI hiển thị.
    Bền với cả doc dạng dict lẫn chuỗi thô.
    """
    evidence: List[Dict[str, Any]] = []
    for rank, d in enumerate(docs, start=1):
        if not isinstance(d, dict):
            evidence.append({
                "rank"           : rank,
                "doc_number"     : "",
                "hierarchy_path" : "",
                "effective_date" : "",
                "content"        : str(d),
                "rerank_score"   : 0.0,
                "graph_boost"    : 0.0,
                "rrf_score"      : 0.0,
            })
            continue

        evidence.append({
            "rank"           : rank,
            "doc_number"     : d.get("doc_number", ""),
            "hierarchy_path" : d.get("hierarchy_path", ""),
            "effective_date" : d.get("effective_date", ""),
            "content"        : (d.get("content") or d.get("text") or ""),
            "rerank_score"   : float(d.get("final_rerank_score") or d.get("score") or 0.0),
            "graph_boost"    : float(d.get("graph_boost", 0.0)),
            "rrf_score"      : float(d.get("rrf_score", 0.0)),
        })
    return evidence


def log_interaction(
    query: str,
    answer: str,
    docs: List[Any],
    mode: str,
    top_k: int,
    model_used: str,
    attribution_score: float,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    is_mock: bool,
    turn_index: int = 0,
    run_id: Optional[str] = None,
) -> bool:
    """
    Ghi một bản ghi đầy đủ của 1 lượt hỏi-đáp vào INTERACTIONS_FILE.
    An toàn: mọi lỗi được nuốt và trả False → KHÔNG bao giờ làm hỏng luồng UI.

    Returns
    -------
    bool — True nếu ghi thành công.
    """
    _ensure_flagging_dir()

    total_ms = float(retrieval_latency_ms) + float(generation_latency_ms)
    record = {
        "timestamp"  : datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id"     : run_id or RUN_ID,
        "turn_index" : turn_index,
        "query"      : query,
        "answer"     : answer,
        "mode"       : mode,
        "top_k"      : top_k,
        "model_used" : model_used,
        "is_mock"    : is_mock,
        "metrics"    : {
            "retrieval_latency_ms"  : round(float(retrieval_latency_ms), 2),
            "generation_latency_ms" : round(float(generation_latency_ms), 2),
            "total_latency_ms"      : round(total_ms, 2),
            "attribution_score"     : round(float(attribution_score), 4),
            "n_docs"                : len(docs),
        },
        "evidence"   : _extract_evidence(docs),
    }

    try:
        with open(INTERACTIONS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Interaction ghi nhận: turn=%d | model=%s | n_docs=%d | query='%.40s...'",
            turn_index, model_used, len(docs), query,
        )
        return True
    except Exception as exc:
        logger.error("Không thể ghi interaction log: %s", exc)
        return False


def load_interactions() -> List[Dict[str, Any]]:
    """Đọc toàn bộ interaction log về list dict (phục vụ phân tích / viết test)."""
    if not INTERACTIONS_FILE.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(INTERACTIONS_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("Không đọc được interaction file: %s", exc)
    return records


def get_interaction_stats() -> Dict[str, Any]:
    """Thống kê nhanh: tổng lượt, độ trễ trung bình, attribution trung bình."""
    recs = load_interactions()
    n = len(recs)
    if n == 0:
        return {
            "total": 0,
            "avg_retrieval_ms": 0.0,
            "avg_generation_ms": 0.0,
            "avg_attribution": 0.0,
        }

    sum_r = sum_g = sum_a = 0.0
    for r in recs:
        m = r.get("metrics", {})
        sum_r += float(m.get("retrieval_latency_ms", 0.0))
        sum_g += float(m.get("generation_latency_ms", 0.0))
        sum_a += float(m.get("attribution_score", 0.0))

    return {
        "total"             : n,
        "avg_retrieval_ms"  : round(sum_r / n, 2),
        "avg_generation_ms" : round(sum_g / n, 2),
        "avg_attribution"   : round(sum_a / n, 4),
    }
